#!/usr/bin/env python3
"""Optional Ray actor supervisor for a Strata layer-stage worker.

Ray starts and monitors the worker on a selected node. Strata's native TCP protocol carries inference activations;
Ray's object store is deliberately not in the model data path.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path


def worker_command(config: dict, bind: str, split: int) -> list[str]:
    exe = config.get("exe")
    args = config.get("args")
    if not isinstance(exe, str) or not exe or not isinstance(args, list) or any(not isinstance(x, str) for x in args):
        raise ValueError('worker config needs a string "exe" and string-list "args"')
    if any(x in args for x in ("--layer-stage-client", "--layer-stage-server", "--layer-stage-split")):
        raise ValueError("remove layer-stage flags from the worker config; the Ray launcher adds the server flags")
    if any(arg == "--gpu" or arg.startswith("--gpu=") for arg in args):
        raise ValueError("remove --gpu from worker args; Strata would override Ray's CUDA_VISIBLE_DEVICES assignment")
    configured_gpu = config.get("gpu")
    if configured_gpu is not None:
        try:
            local_gpu = int(configured_gpu)
        except (TypeError, ValueError) as exc:
            raise ValueError('worker config "gpu" must be local visible index 0') from exc
        if local_gpu != 0:
            raise ValueError("Ray assigned one visible GPU; worker config gpu must be local index 0")
    run_args = list(args)
    if "--serve" not in run_args:
        run_args.insert(0, "--serve")
    return [exe, *run_args, "--layer-stage-server", bind, "--layer-stage-split", str(split)]


def worker_environment(config: dict, token: str, inherited: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if inherited is None else inherited)
    configured = config.get("env") or {}
    if not isinstance(configured, dict):
        raise ValueError('worker config "env" must be an object')
    if any(str(k).upper() == "CUDA_VISIBLE_DEVICES" for k in configured):
        raise ValueError('worker config "env" cannot override Ray-assigned CUDA_VISIBLE_DEVICES')
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip() or visible.strip() == "-1":
        raise ValueError("Ray did not assign a visible CUDA device to the layer-stage actor")
    visible_devices = [part.strip() for part in visible.strip().split(",")]
    if len(visible_devices) != 1 or not visible_devices[0]:
        raise ValueError(f"Ray assigned {len(visible_devices)} CUDA devices; the worker actor requires exactly one")
    env.update({str(k): str(v) for k, v in configured.items()})
    lib_dirs = config.get("lib_dirs") or []
    if not isinstance(lib_dirs, list) or any(not isinstance(p, str) for p in lib_dirs):
        raise ValueError('worker config "lib_dirs" must be a string-list')
    if lib_dirs:
        key = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[key] = os.pathsep.join([*lib_dirs, env.get(key, "")]).rstrip(os.pathsep)
    env["STRATA_RPC_TOKEN"] = token
    return env


def ray_get_bounded(ray, ref, timeout: float, operation: str):
    """Wait before get: Ray's get timeout does not bound every placement/startup state consistently."""
    ready, _ = ray.wait([ref], timeout=max(0.0, timeout), num_returns=1)
    if not ready:
        raise TimeoutError(f"timed out waiting for Ray {operation}")
    return ray.get(ref)


def worker_node_id(nodes: list[dict], address: str) -> str:
    matches = [n for n in nodes if n.get("Alive") and n.get("NodeManagerAddress") == address and n.get("NodeID")]
    if len(matches) != 1:
        raise ValueError(f"expected one live Ray node at {address}, found {len(matches)}")
    return matches[0]["NodeID"]


def make_worker_actor(ray):
    @ray.remote(num_gpus=1, max_concurrency=2)
    class LayerStageWorker:
        def __init__(self):
            self.proc = None
            self.lines = deque(maxlen=2000)
            self.line_count = 0
            self.lock = threading.Lock()
            self.proc_lock = threading.Lock()
            self.stop_lock = threading.Lock()
            self.ready = threading.Event()
            self.reader = None

        def start(self, config_path: str, bind: str, split: int, token: str, timeout: float):
            path = Path(config_path)
            config = json.loads(path.read_text(encoding="utf-8-sig"))
            command = worker_command(config, bind, split)
            env = worker_environment(config, token)
            cwd = config.get("cwd")
            with self.proc_lock:
                self.proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                             bufsize=1)
                proc = self.proc

            def read_output():
                assert proc.stdout is not None
                try:
                    for raw in proc.stdout:
                        line = raw.rstrip("\r\n")
                        with self.lock:
                            self.lines.append(line)
                            self.line_count += 1
                        if line.startswith("LAYER_STAGE_READY "):
                            self.ready.set()
                finally:
                    self.ready.set()

            self.reader = threading.Thread(target=read_output, name="strata-stage-worker-output", daemon=True)
            try:
                self.reader.start()
                deadline = time.monotonic() + timeout
                while not self.ready.wait(min(0.25, max(0.0, deadline - time.monotonic()))):
                    if proc.poll() is not None:
                        raise RuntimeError("Strata worker exited before LAYER_STAGE_READY")
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Strata worker did not become ready in {timeout:.0f} seconds")
                if proc.poll() is not None:
                    raise RuntimeError("Strata worker exited during startup")
                with self.lock:
                    ready = any(line.startswith("LAYER_STAGE_READY ") for line in self.lines)
                if not ready:
                    raise RuntimeError("Strata worker output closed before LAYER_STAGE_READY")
                return True
            except BaseException:
                self.stop()
                raise

        def tail(self, start: int = 0):
            with self.lock:
                lines = list(self.lines)
                total = self.line_count
                first = total - len(lines)
            offset = max(0, start - first)
            return lines[offset:], total, None if self.proc is None else self.proc.poll()

        def stop(self):
            with self.stop_lock:
                with self.proc_lock:
                    proc = self.proc
                if proc is None:
                    return
                if proc.poll() is None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired as exc:
                            raise RuntimeError("Strata worker did not exit after terminate and kill") from exc
                reader = self.reader
                if reader is not None and reader.is_alive() and reader is not threading.current_thread():
                    reader.join(timeout=2)
                if reader is not None and reader.is_alive() and proc.stdout is not None:
                    proc.stdout.close()
                    reader.join(timeout=1)

    return LayerStageWorker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-address", default="auto", help='Ray cluster address (default: "auto")')
    parser.add_argument("--worker-node-ip", required=True, help="NodeManagerAddress of the 5070 Ti Ray node")
    parser.add_argument("--worker-config", required=True,
                        help="path to the 5070 Ti's local Strata run-config JSON (the path is read on that node)")
    parser.add_argument("--bind", default="0.0.0.0:39001", help="worker listen address; default: 0.0.0.0:39001")
    parser.add_argument("--advertise-ip", help="address the 4090 can reach; defaults to --worker-node-ip")
    parser.add_argument("--split", type=int, required=True, help="first model layer owned by the worker")
    parser.add_argument("--ready-timeout", type=float, default=900.0, help="worker startup limit in seconds")
    parser.add_argument("--token-env", default="STRATA_RPC_TOKEN", help="environment variable holding the RPC secret")
    args = parser.parse_args()
    token = os.environ.get(args.token_env, "")
    if not token:
        parser.error(f"set {args.token_env} to the same long random secret on the host and worker")
    if args.split < 2 or args.ready_timeout <= 0:
        parser.error("--split must be at least 2 and --ready-timeout must be positive")
    if ":" not in args.bind:
        parser.error("--bind must be ADDRESS:PORT")

    try:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    except ImportError as e:
        print(f"Ray is optional and is not installed in this Python environment: {e}", file=sys.stderr)
        return 2

    ray.init(address=args.ray_address)
    ready_timeout = args.ready_timeout
    startup_deadline = time.monotonic() + ready_timeout
    actor = None
    start_ref = None
    try:
        node_id = worker_node_id(ray.nodes(), args.worker_node_ip)
        actor_type = make_worker_actor(ray)
        actor = actor_type.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        ).remote()
        # This deadline includes actor placement and GPU resource scheduling, not just worker model loading.
        start_ref = actor.start.remote(args.worker_config, args.bind, args.split, token, ready_timeout)
        offset = 0
        tail_ref = actor.tail.remote(offset)
        while True:
            remaining = startup_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Ray actor placement and worker startup exceeded {ready_timeout:.0f} seconds")
            ready, _ = ray.wait([start_ref, tail_ref], timeout=min(2.0, remaining), num_returns=1)
            if not ready:
                continue
            ref = ready[0]
            value = ray.get(ref)  # ray.wait has established completion; this does not wait on placement or execution.
            if ref == start_ref:
                break
            lines, offset, code = value
            for line in lines:
                print(f"[5070 Ti] {line}", flush=True)
            if code is not None:
                raise RuntimeError(f"Strata worker exited during startup with code {code}")
            tail_ref = actor.tail.remote(offset)  # never queue more than one tail call.
        host = args.advertise_ip or args.worker_node_ip
        port = args.bind.rsplit(":", 1)[1]
        print("\nStrata layer-stage worker is ready. Add these arguments to the 4090 run config:", flush=True)
        print(f'  "--layer-stage-client", "{host}:{port}",', flush=True)
        print(f'  "--layer-stage-split", "{args.split}"', flush=True)
        print("The 4090 API process still runs Strata normally. Keep this supervisor running; Ctrl+C stops the worker.",
              flush=True)
        while True:
            if tail_ref is None:
                tail_ref = actor.tail.remote(offset)
            try:
                lines, offset, code = ray_get_bounded(ray, tail_ref, 5.0, "worker monitor")
            except TimeoutError:
                continue  # polling delay, not evidence that the worker or actor failed; keep one tail future.
            tail_ref = None
            for line in lines:
                print(f"[5070 Ti] {line}", flush=True)
            if code is not None:
                raise RuntimeError(f"Strata worker exited with code {code}")
    except KeyboardInterrupt:
        print("Stopping the remote Strata layer-stage worker...", flush=True)
    except Exception as e:
        print(f"Ray layer-stage worker failed: {e}", file=sys.stderr)
        return 1
    finally:
        if actor is not None:
            try:
                ray_get_bounded(ray, actor.stop.remote(), 15.0, "worker shutdown")
            except Exception as e:
                print(f"Warning: bounded worker cleanup did not complete: {e}", file=sys.stderr)
        shutdown_thread = threading.Thread(target=ray.shutdown, name="ray-shutdown", daemon=True)
        shutdown_thread.start()
        shutdown_thread.join(timeout=5)
        if shutdown_thread.is_alive():
            print("Warning: Ray shutdown exceeded 5 seconds; worker stop was already bounded separately", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
