#!/usr/bin/env python3
"""Validate the Windows RPC archive layout and every packaged file checksum."""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.layer_stage_ray import worker_command, worker_environment


def documented_configs(guide: str) -> list[dict]:
    lines = guide.splitlines()
    configs = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("    {"):
            block = [lines[i][4:]]
            i += 1
            while i < len(lines):
                block.append(lines[i][4:] if lines[i].startswith("    ") else lines[i])
                if lines[i] == "    }":
                    break
                i += 1
            configs.append(json.loads("\n".join(block)))
        i += 1
    return configs


def validate(archive: Path) -> None:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        required = {
            "strata.exe",
            "layer_stage_ray.py",
            "RPC.md",
            "LAYER_STAGE_RPC.md",
            "BUILD.json",
            "SHA256SUMS.txt",
        }
        missing = required - names
        if missing:
            raise AssertionError(f"Windows package is missing root files: {sorted(missing)}")
        if "tools/layer_stage_ray.py" in names:
            raise AssertionError("Ray script must be at archive root to match extraction into C:\\Strata\\engine")

        sums = zf.read("SHA256SUMS.txt").decode("ascii").splitlines()
        recorded: dict[str, str] = {}
        for line in sums:
            digest, name = line.split("  ", 1)
            if name in recorded:
                raise AssertionError(f"duplicate checksum entry: {name}")
            recorded[name] = digest.lower()
        packaged = names - {"SHA256SUMS.txt"}
        if set(recorded) != packaged:
            raise AssertionError("SHA256SUMS.txt does not cover exactly the packaged files")
        for name, expected in recorded.items():
            digest = hashlib.sha256()
            with zf.open(name) as source:
                for block in iter(lambda: source.read(1 << 20), b""):
                    digest.update(block)
            actual = digest.hexdigest()
            if actual != expected:
                raise AssertionError(f"package checksum mismatch: {name}")

        guide = zf.read("LAYER_STAGE_RPC.md").decode("utf-8")
        if "C:\\Strata\\engine" not in guide or ".\\engine\\layer_stage_ray.py" not in guide:
            raise AssertionError("layer-stage guide does not match the package extraction/script path")
        configs = documented_configs(guide)
        if len(configs) != 2:
            raise AssertionError(f"expected worker and coordinator JSON examples, found {len(configs)}")
        worker, coordinator = configs
        worker_args = worker["args"]
        coordinator_args = coordinator["args"]
        if worker_args[worker_args.index("--prefill") + 1] != "2048" or "--gpu" in worker_args:
            raise AssertionError("worker example must use fixed prefill and preserve Ray's GPU visibility")
        if coordinator_args[coordinator_args.index("--prefill") + 1] != "2048":
            raise AssertionError("coordinator example must use the same fixed prefill chunk")
        shared_args = ("--pack", "--native", "--ple-gguf", "--expert-profile", "--expert-cache", "--prefill",
                       "--spec", "--max-context", "--kv")
        for option in shared_args:
            worker_value = worker_args[worker_args.index(option) + 1]
            coordinator_value = coordinator_args[coordinator_args.index(option) + 1]
            if worker_value != coordinator_value:
                raise AssertionError(f"worker and coordinator examples differ at {option}")
        unsupported = {"--spec-min-p", "--mtp", "--vision", "--kv-resident"}
        if unsupported.intersection(worker_args) or unsupported.intersection(coordinator_args):
            raise AssertionError("example includes an option unsupported by layer-stage mode")
        if coordinator.get("sampling", {}).get("temperature") != 0:
            raise AssertionError("coordinator example must default to greedy sampling")
        if not coordinator.get("tokenizer") or not coordinator.get("model_name"):
            raise AssertionError("coordinator example is missing the API tokenizer or model name")
        if coordinator_args[coordinator_args.index("--layer-stage-split") + 1] != "24":
            raise AssertionError("coordinator example split must match the documented worker example")
        command = worker_command(worker, "0.0.0.0:39001", 24)
        if "--serve" not in command or "--layer-stage-server" not in command:
            raise AssertionError("worker example does not produce a serving worker command")
        worker_environment(worker, "package-test", {"CUDA_VISIBLE_DEVICES": "GPU-package-test"})


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python tests/test_rpc_windows_package.py path-to-archive.zip")
    validate(Path(sys.argv[1]))
    print("Windows RPC archive contents, layout, documentation path, and checksums passed")
