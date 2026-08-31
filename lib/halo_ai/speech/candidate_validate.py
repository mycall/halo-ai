#!/usr/bin/env python3
"""Validate the isolated ROCm 10 speech candidate without loading model weights."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any


EXPECTED_PYTHON = "3.14.7"
EXPECTED_HIP = "10.0.0"
EXPECTED_DISTRIBUTIONS = {
    "accelerate": "1.14.0",
    "fastapi": "0.141.1",
    "gradio": "6.26.0",
    "numpy": "2.5.2",
    "protobuf": "7.36.0",
    "safetensors": "0.8.0",
    "scipy": "1.18.1",
    "sentencepiece": "0.2.2",
    "soundfile": "0.14.0",
    "tiktoken": "0.14.0",
    "torch": "2.13.0+rocm10.0.0",
    "torchaudio": "2.11.0.2+rocm10.0.0",
    "torchvision": "0.28.0+rocm10.0.0",
    "transformers": "5.16.1",
    "uvicorn": "0.52.4",
}
REQUIRED_IMPORTS = (
    "accelerate",
    "fastapi",
    "google.protobuf",
    "gradio",
    "multipart",
    "numpy",
    "safetensors",
    "scipy",
    "sentencepiece",
    "soundfile",
    "tiktoken",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "uvicorn",
)
LOCK_FILES = (
    "requirements-rocm10-py314.lock",
    "requirements-application-py314.lock",
)


def locked_distributions() -> dict[str, str]:
    expected: dict[str, str] = {}
    for name in LOCK_FILES:
        path = Path(__file__).with_name(name)
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.count("==") != 1:
                raise RuntimeError(f"non-exact requirement in {path.name}: {line}")
            distribution, version = line.split("==", 1)
            if distribution in expected:
                raise RuntimeError(f"duplicate locked distribution: {distribution}")
            expected[distribution] = version
    return expected


def validate(*, require_gpu: bool) -> dict[str, Any]:
    locked = locked_distributions()
    direct_lock_mismatches = {
        name: {"expected": expected, "locked": locked.get(name)}
        for name, expected in EXPECTED_DISTRIBUTIONS.items()
        if locked.get(name) != expected
    }
    if direct_lock_mismatches:
        raise RuntimeError(
            "candidate direct-version lock mismatch: "
            f"{json.dumps(direct_lock_mismatches, sort_keys=True)}"
        )
    versions = {
        name: importlib.metadata.version(name)
        for name in locked
    }
    mismatches = {
        name: {"expected": expected, "actual": versions.get(name)}
        for name, expected in locked.items()
        if versions.get(name) != expected
    }
    python_version = platform.python_version()
    if python_version != EXPECTED_PYTHON:
        mismatches["python"] = {
            "expected": EXPECTED_PYTHON,
            "actual": python_version,
        }
    imported = []
    for name in REQUIRED_IMPORTS:
        importlib.import_module(name)
        imported.append(name)

    torch = importlib.import_module("torch")
    hip_version = torch.version.hip
    if hip_version != EXPECTED_HIP:
        mismatches["hip"] = {"expected": EXPECTED_HIP, "actual": hip_version}
    if mismatches:
        raise RuntimeError(f"candidate version mismatch: {json.dumps(mismatches, sort_keys=True)}")

    result: dict[str, Any] = {
        "status": "packages-valid",
        "python": python_version,
        "hip": hip_version,
        "versions": versions,
        "imports": imported,
        "gpu_required": require_gpu,
    }
    if not require_gpu:
        return result
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device is unavailable through torch.cuda")
    architectures = torch.cuda.get_arch_list()
    if not any(item.startswith("gfx1151") for item in architectures):
        raise RuntimeError(f"PyTorch does not advertise gfx1151: {architectures}")
    properties = torch.cuda.get_device_properties(0)
    device_architecture = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if not device_architecture.startswith("gfx1151"):
        raise RuntimeError(
            f"ROCm device is not gfx1151: {device_architecture or properties.name}"
        )
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    product = left @ left
    torch.cuda.synchronize()
    actual = product.cpu().tolist()
    expected = [[7.0, 10.0], [15.0, 22.0]]
    if actual != expected:
        raise RuntimeError(f"unexpected GPU matrix result: {actual}")
    result.update({
        "status": "gpu-valid",
        "device": properties.name,
        "device_architecture": device_architecture,
        "architectures": architectures,
        "matrix_result": actual,
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="require and exercise gfx1151")
    args = parser.parse_args()
    print(json.dumps(validate(require_gpu=args.gpu), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
