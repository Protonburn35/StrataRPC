#!/usr/bin/env python3
"""Package a portable Windows CUDA engine for the experimental RPC release."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import sysconfig
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA_PACKAGES = {"nvidia-cublas", "nvidia-cuda-runtime"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_nvidia_licenses(stage: Path, nvidia_root: Path) -> None:
    licenses = stage / "licenses" / "nvidia"
    copied = 0
    for path in nvidia_root.rglob("*"):
        if path.is_file() and any(key in path.name.lower() for key in ("license", "notice", "eula", "copying")):
            relative = path.relative_to(nvidia_root)
            target = licenses / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
    for dist in importlib.metadata.distributions():
        if (dist.metadata.get("Name") or "").lower() not in CUDA_PACKAGES:
            continue
        for item in dist.files or ():
            name = Path(str(item)).name.lower()
            if not any(key in name for key in ("license", "notice", "eula", "copying")):
                continue
            source = Path(dist.locate_file(item))
            if not source.is_file():
                continue
            target = licenses / dist.metadata["Name"] / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
    if copied == 0:
        raise RuntimeError("NVIDIA CUDA wheel license files were not found")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cuda-version", required=True)
    parser.add_argument("--archs", default="89,120")
    args = parser.parse_args()

    exe = args.exe.resolve()
    if not exe.is_file():
        parser.error(f"engine executable does not exist: {exe}")
    nvidia_root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    if not nvidia_root.is_dir():
        raise RuntimeError(f"NVIDIA CUDA Python wheels are missing under {nvidia_root}")
    dlls = sorted(nvidia_root.rglob("*.dll"))
    required = {"cublas64_13.dll", "cudart64_13.dll"}
    available = {path.name.lower() for path in dlls}
    if not required <= available:
        raise RuntimeError(f"CUDA runtime wheel is missing required DLLs: {sorted(required - available)}")

    version_match = re.search(r"project\(strata VERSION ([0-9.]+)", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8"))
    if version_match is None:
        raise RuntimeError("could not determine Strata version from CMakeLists.txt")
    archs = sorted({int(value) for value in args.archs.split(",") if value.strip()})
    if not archs:
        raise RuntimeError("at least one CUDA architecture is required")

    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / "strata-rpc-windows-x64.zip"
    with tempfile.TemporaryDirectory(prefix="strata-rpc-package-") as temporary:
        stage = Path(temporary)
        shutil.copy2(exe, stage / "strata.exe")
        shutil.copy2(ROOT / "LICENSE", stage / "LICENSE")
        shutil.copy2(ROOT / "docs" / "RPC.md", stage / "RPC.md")
        shutil.copy2(ROOT / "docs" / "LAYER_STAGE_RPC.md", stage / "LAYER_STAGE_RPC.md")
        shutil.copy2(ROOT / "tools" / "layer_stage_ray.py", stage / "layer_stage_ray.py")
        (stage / "BUILD.json").write_text(json.dumps({
            "source": "prebuilt",
            "backend": "cuda",
            "platform": "windows-x64",
            "version": version_match.group(1),
            "archs": archs,
            "vision": "none",
            "toolkit": 13,
            "cuda_version": args.cuda_version,
        }, indent=1) + "\n", encoding="utf-8")

        # These are the same CUDA 13 runtime and cuBLAS wheels Strata setup installs for a prebuilt CUDA 13 engine.
        names: set[str] = set()
        for dll in dlls:
            name = dll.name.lower()
            if name in names:
                raise RuntimeError(f"duplicate CUDA DLL name in installed wheels: {dll.name}")
            names.add(name)
            shutil.copy2(dll, stage / dll.name)
        copy_nvidia_licenses(stage, nvidia_root)

        checksums = []
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            digest = sha256_file(path)
            checksums.append(f"{digest}  {path.relative_to(stage).as_posix()}")
        (stage / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")

        archive.unlink(missing_ok=True)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
            for path in sorted(p for p in stage.rglob("*") if p.is_file()):
                output.write(path, path.relative_to(stage).as_posix())
    print(f"Created {archive} ({archive.stat().st_size / (1024 * 1024):.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
