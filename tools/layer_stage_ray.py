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
    run_args = list(args)
    if "--serve" not in run_args:
        run_args.insert(0, "--serve")
    return [exe, *run_args, "--layer-stage-server", bind, "--layer-stage-split", str(split)]


def worker_environment(config: dict, token: str, inherited: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if inherited is None else inherited)
    configured = config.get("env") or {}
    if not isinstance(configured, dict):
        raise ValueError('worker config "env" must be an object')
    env.update({str(k): str(v) for k, v in configured.items()})
    lib_dirs = config.get("lib_dirs") or []
    if not isinstance(lib_dirs, list) or any(not isinstance(p, str) for p in lib_dirs):
        raise ValueError('worker config "lib_dirs" must be a string-list')
    if lib_dirs:
        key = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[key] = os.pathsep.join([*lib_dirs, env.get(key, "")]).rstrip(os.pathsep)
    env["STRATA_RPC_TOKEN"] = token
    return env


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
            self.ready = threading.Event()
            self.reader = None

        def start(self, config_path: str, bind: str, split: int, token: str, timeout: float):
            path = Path(config_path)
            config = json.loads(path.read_text(encoding="utf-8-sig"))
            command = worker_command(config, bind, split)
            env = worker_environment(config, token)
            cwd = config.get("cwd")
            self.proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                         bufsize=1)

            def read_output():
                assert self.proc is not None and self.proc.stdout is not None
                try:
                    for raw in self.proc.stdout:
                        line = raw.rstrip("\r\n")
                        with self.lock:
                            self.lines.append(line)
                            self.line_count += 1
                        if line.startswith("LAYER_STAGE_READY "):
                            self.ready.set()
                finally:
                    self.ready.set()

            self.reader = threading.Thread(target=read_output, name="strata-stage-worker-output", daemon=True)
            self.reader.start()
            deadline = time.monotonic() + timeout
            while not self.ready.wait(0.25):
                if self.proc.poll() is not None:
                    raise RuntimeError("Strata worker exited before LAYER_STAGE_READY")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Strata worker did not become ready in {timeout:.0f} seconds")
            if self.proc.poll() is not None:
                raise RuntimeError("Strata worker exited during startup")
            with self.lock:
                ready = any(line.startswith("LAYER_STAGE_READY ") for line in self.lines)
            if not ready:
                raise RuntimeError("Strata worker output closed before LAYER_STAGE_READY")
            return True

        def tail(self, start: int = 0):
            with self.lock:
                lines = list(self.lines)
                total = self.line_count
                first = total - len(lines)
            offset = max(0, start - first)
            return lines[offset:], total, None if self.proc is None else self.proc.poll()

        def stop(self):
            proc = self.proc
            if proc is None or proc.poll() is not None:
                return
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

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
    actor = None
    start_ref = None
    try:
        node_id = worker_node_id(ray.nodes(), args.worker_node_ip)
        actor_type = make_worker_actor(ray)
        actor = actor_type.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        ).remote()
        start_ref = actor.start.remote(args.worker_config, args.bind, args.split, token, args.ready_timeout)
        offset = 0
        while True:
            ready, _ = ray.wait([start_ref], timeout=5.0)
            try:
                lines, offset, code = ray.get(actor.tail.remote(offset))
                for line in lines:
                    print(f"[5070 Ti] {line}", flush=True)
            except Exception:
                code = None
            if code is not None and not ready:
                raise RuntimeError(f"Strata worker exited during startup with code {code}")
            if ready:
                ray.get(start_ref)
                break
        host = args.advertise_ip or args.worker_node_ip
        port = args.bind.rsplit(":", 1)[1]
        print("\nStrata layer-stage worker is ready. Add these arguments to the 4090 run config:", flush=True)
        print(f'  "--layer-stage-client", "{host}:{port}",', flush=True)
        print(f'  "--layer-stage-split", "{args.split}"', flush=True)
        print("The 4090 API process still runs Strata normally. Keep this supervisor running; Ctrl+C stops the worker.",
              flush=True)
        while True:
            time.sleep(5)
            lines, offset, code = ray.get(actor.tail.remote(offset))
            for line in lines:
                print(f"[5070 Ti] {line}", flush=True)
            if code is not None:
                raise RuntimeError(f"Strata worker exited with code {code}")
    except KeyboardInterrupt:
        print("Stopping the remote Strata layer-stage worker...", flush=True)
    except Exception as e:
        print(f"Ray layer-stage worker failed: {e}", file=sys.stderr)
        if actor is not None:
            try:
                lines, _, _ = ray.get(actor.tail.remote(0), timeout=5)
                for line in lines[-60:]:
                    print(f"[5070 Ti] {line}", file=sys.stderr)
            except Exception:
                pass
        return 1
    finally:
        if actor is not None:
            try:
                ray.get(actor.stop.remote(), timeout=15)
            except Exception:
                pass
        ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
