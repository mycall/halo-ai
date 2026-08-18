#!/usr/bin/env python3
"""Safe lifecycle manager for local Strix Halo inference services."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import copy
import fcntl
import hashlib
import json
import os
import pwd
import re
import shlex
import shutil
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, NoReturn

import host_profile
import longbench
import rocmfpx_tune


VERSION = "0.1.0"
MANAGED_LABEL = "local.halo-ai.managed=true"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = PROJECT_ROOT / "config"
ENGINE_IMAGE_KEYS = {
    "lemonade": "LEMONADE_IMAGE",
    "llamacpp": "LLAMACPP_IMAGE",
    "rocmfpx": "ROCMFPX_IMAGE",
    "ds4": "DS4_IMAGE",
    "speech": "SPEECH_IMAGE",
    "vllm": "VLLM_IMAGE",
}
ENGINE_PORT_KEYS = {
    "lemonade": "LEMONADE_PORT",
    "llamacpp": "LLAMACPP_PORT",
    "rocmfpx": "ROCMFPX_PORT",
    "ds4": "DS4_PORT",
    "speech": "SPEECH_PORT",
    "vllm": "VLLM_PORT",
}
ENGINE_CONTAINERS = {
    "lemonade": "halo-lemonade",
    "llamacpp": "halo-llamacpp",
    "rocmfpx": "halo-rocmfpx",
    "ds4": "halo-ds4",
    "speech": "halo-speech",
    "vllm": "halo-vllm",
}

ROCMFPX_ENGINE_URL = (
    "https://github.com/julianmb/q38rocm/releases/download/v1.0.0/"
    "strix-halo-rocmfpx-engine-v1.0.0-linux-x86_64.tar.gz"
)
ROCMFPX_ENGINE_BYTES = 55_255_915
ROCMFPX_ENGINE_SHA256 = "bbc7845db0c012b97f1c9b8a2733a7083c6f9a749a453866fbe1994151d3364f"
ROCMFPX_Q38ROCM_COMMIT = "66de2f3bc625249eabff5bd919fd6dbdd3d7ccaa"
ROCMFPX_VULKAN_BASE = "sha256:8cdde6d42ab621b3d0f1e02618f4234c58b7036bec984a2a64fa205d38f99922"
ROCMFPX_ROCM_BASE = "sha256:32d25e6f7608e1d221b71f51389c883afc655b9a3add9f7a787453dca288117b"


class HaloError(RuntimeError):
    """A user-facing, fail-closed error."""


def fail(message: str) -> NoReturn:
    raise HaloError(message)


def eprint(*values: object) -> None:
    print(*values, file=sys.stderr)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")


def atomic_write(path: Path, data: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def atomic_json(path: Path, value: Any, mode: int = 0o600) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n", mode)


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE data without evaluating shell syntax."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for number, raw in enumerate(read_text(path).splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Z][A-Z0-9_]*)\s*=\s*(.*)", line)
        if not match:
            fail(f"invalid configuration line {path}:{number}")
        key, value = match.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if "\x00" in value or "\n" in value:
            fail(f"invalid control character in {path}:{number}")
        result[key] = value
    return result


DEFAULTS = {
    "HALO_AI_RUN_USER": "",
    "HALO_AI_STATE_DIR": "/var/opt/halo-ai/state",
    "HALO_AI_CACHE_DIR": "/var/cache/halo-ai",
    "HALO_AI_MODELS_ROOT": "/srv/halo-ai/models",
    "HALO_AI_CATALOG_DIR": "/etc/opt/halo-ai/models.d",
    "HALO_AI_PRESET_DIR": "/etc/opt/halo-ai/request-presets.d",
    "HALO_AI_INVENTORY_FILE": "/var/opt/halo-ai/state/model-inventory.json",
    "HALO_AI_MODELS_REQUIRE_MOUNT": "0",
    "HALO_AI_MODELS_EXPECT_UUID": "",
    "HALO_AI_DEFAULT_PROFILE": "qwen3.6-35b-a3b-q8xl-lemonade",
    "LEMONADE_IMAGE": "ghcr.io/lemonade-sdk/lemonade-server:latest",
    "LEMONADE_PORT": "13305",
    "LEMONADE_VALIDATION_MODEL": "Qwen3-0.6B-GGUF",
    "LEMONADE_LLAMACPP_ROCM_BIN": "latest",
    "LLAMACPP_IMAGE": "docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.14",
    "LLAMACPP_PORT": "8080",
    "ROCMFPX_IMAGE": "localhost/halo-ai-rocmfpx:v1.0.0",
    "ROCMFPX_PORT": "8002",
    "DS4_IMAGE": "docker.io/kyuz0/strix-halo-ds4-toolbox:rocm-7.14",
    "DS4_PORT": "8000",
    "DS4_KV_CACHE_ENABLED": "0",
    "DS4_KV_CACHE_DIR": "/var/cache/halo-ai/ds4-kv",
    "DS4_KV_CACHE_MB": "8192",
    "SPEECH_IMAGE": "localhost/halo-ai-speech:rocm-7.14",
    "SPEECH_PORT": "7860",
    "SPEECH_TEST_AUDIO_PATH": "/var/cache/halo-ai/speech/input1.wav",
    "VLLM_IMAGE": "",
    "VLLM_PORT": "8001",
    "VLLM_GGUF_EXPERIMENTAL": "0",
    "HALO_AI_GTT_TARGET_GIB": "auto",
    "HALO_AI_GTT_AUTOTUNE": "stage",
    "HALO_AI_GTT_CANDIDATES_GIB": "112,116,118",
    "HALO_AI_GTT_MAX_GIB": "118",
    "HALO_AI_OS_RESERVE_GIB": "4",
}


@dataclasses.dataclass(frozen=True)
class Config:
    values: dict[str, str]
    files: tuple[Path, ...]

    def get(self, key: str) -> str:
        return self.values[key]

    def path(self, key: str) -> Path:
        value = self.get(key)
        path = Path(value)
        if not path.is_absolute() or path == Path("/"):
            fail(f"{key} must be a safe absolute path, got {value!r}")
        return path

    def integer(self, key: str, minimum: int = 0, maximum: int = 2**31) -> int:
        value = self.get(key)
        if not re.fullmatch(r"[0-9]+", value):
            fail(f"{key} must be numeric, got {value!r}")
        number = int(value)
        if not minimum <= number <= maximum:
            fail(f"{key} must be in [{minimum}, {maximum}], got {number}")
        return number

    def boolean(self, key: str) -> bool:
        value = self.get(key)
        if value not in {"0", "1"}:
            fail(f"{key} must be 0 or 1, got {value!r}")
        return value == "1"


def load_config(explicit: str | None) -> Config:
    values = dict(DEFAULTS)
    loaded: list[Path] = []
    system = Path("/etc/opt/halo-ai/config.env")
    user = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "halo-ai/config.env"
    candidates = [system, user]
    if explicit:
        selected = Path(explicit).expanduser()
        if not selected.is_absolute():
            selected = (Path.cwd() / selected).resolve()
        candidates = [selected]
    for path in candidates:
        if path.exists():
            values.update(parse_env_file(path))
            loaded.append(path)
    if not Path(values["HALO_AI_CATALOG_DIR"]).exists():
        values["HALO_AI_CATALOG_DIR"] = str(SOURCE_CONFIG / "models.d")
    if not Path(values["HALO_AI_PRESET_DIR"]).exists():
        values["HALO_AI_PRESET_DIR"] = str(SOURCE_CONFIG / "request-presets.d")
    config = Config(values, tuple(loaded))
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    for key in (
        "HALO_AI_STATE_DIR", "HALO_AI_CACHE_DIR", "HALO_AI_MODELS_ROOT",
        "HALO_AI_CATALOG_DIR", "HALO_AI_PRESET_DIR", "HALO_AI_INVENTORY_FILE",
        "DS4_KV_CACHE_DIR", "SPEECH_TEST_AUDIO_PATH",
    ):
        config.path(key)
    for key in ("LEMONADE_PORT", "LLAMACPP_PORT", "ROCMFPX_PORT", "DS4_PORT", "SPEECH_PORT", "VLLM_PORT"):
        config.integer(key, 1, 65535)
    config.integer("DS4_KV_CACHE_MB", 1024, 1024 * 1024)
    for key in ("HALO_AI_MODELS_REQUIRE_MOUNT", "DS4_KV_CACHE_ENABLED", "VLLM_GGUF_EXPERIMENTAL"):
        config.boolean(key)
    candidates_text = config.get("HALO_AI_GTT_CANDIDATES_GIB")
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", candidates_text):
        fail("HALO_AI_GTT_CANDIDATES_GIB must be comma-separated whole GiB values")
    candidates = [int(value) for value in candidates_text.split(",")]
    if candidates != sorted(set(candidates)):
        fail("HALO_AI_GTT_CANDIDATES_GIB must be strictly increasing")
    maximum = config.integer("HALO_AI_GTT_MAX_GIB", 1, 127)
    reserve = config.integer("HALO_AI_OS_RESERVE_GIB", 4, 64)
    if candidates[-1] > maximum or maximum > 128 - reserve:
        fail("GTT candidates/maximum violate the configured 128 GiB OS reserve")
    target = config.get("HALO_AI_GTT_TARGET_GIB")
    if target != "auto":
        if not target.isdigit() or int(target) not in candidates:
            fail("HALO_AI_GTT_TARGET_GIB must be auto or an approved candidate")
    if config.get("HALO_AI_GTT_AUTOTUNE") not in {"observe", "suggest", "stage"}:
        fail("HALO_AI_GTT_AUTOTUNE must be observe, suggest, or stage")
    rocm_bin = config.get("LEMONADE_LLAMACPP_ROCM_BIN")
    if not re.fullmatch(r"(?:builtin|latest|b[0-9]+)", rocm_bin):
        fail("LEMONADE_LLAMACPP_ROCM_BIN must be builtin, latest, or a bNNNN build")


def assert_operator(config: Config) -> None:
    configured = config.get("HALO_AI_RUN_USER")
    if not configured:
        return
    try:
        account = pwd.getpwnam(configured)
    except KeyError:
        fail(f"HALO_AI_RUN_USER does not exist: {configured}")
    if account.pw_uid == 0:
        fail("HALO_AI_RUN_USER must not be root")
    if os.geteuid() != account.pw_uid:
        fail(f"run this lifecycle command as {configured}, not UID {os.geteuid()}")


@dataclasses.dataclass
class Catalog:
    models: dict[str, dict[str, Any]]
    profiles: dict[str, dict[str, Any]]
    sources: list[Path]


def load_catalog(config: Config) -> Catalog:
    directory = config.path("HALO_AI_CATALOG_DIR")
    files = sorted(directory.glob("*.json"))
    if not files:
        fail(f"no catalog JSON files found in {directory}")
    models: dict[str, dict[str, Any]] = {}
    profiles: dict[str, dict[str, Any]] = {}
    for path in files:
        try:
            document = json.loads(read_text(path))
        except json.JSONDecodeError as exc:
            fail(f"invalid JSON in {path}: {exc}")
        if document.get("schema_version") != 1:
            fail(f"unsupported catalog schema in {path}")
        for kind, destination in (("models", models), ("profiles", profiles)):
            for item in document.get(kind, []):
                identifier = item.get("id")
                if not isinstance(identifier, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]+", identifier):
                    fail(f"invalid {kind[:-1]} ID in {path}: {identifier!r}")
                if identifier in destination:
                    fail(f"duplicate {kind[:-1]} ID {identifier}")
                destination[identifier] = item
    validate_catalog(models, profiles)
    return Catalog(models, profiles, files)


def validate_catalog(models: dict[str, dict[str, Any]], profiles: dict[str, dict[str, Any]]) -> None:
    roles = {"main", "mmproj", "dspark", "chat_template", "weights", "processor"}
    for identifier, model in models.items():
        engines = model.get("engines")
        files = model.get("files")
        model_format = model.get("format", "gguf")
        if model_format not in {"gguf", "transformers"}:
            fail(f"model {identifier} has unsupported format {model_format!r}")
        if not isinstance(engines, list) or not engines or not isinstance(files, list) or not files:
            fail(f"model {identifier} requires engines and files")
        if any(engine not in ENGINE_IMAGE_KEYS for engine in engines):
            fail(f"model {identifier} names an unsupported engine")
        if model_format == "transformers" and engines != ["speech"]:
            fail(f"Transformers model {identifier} is only approved for the speech engine")
        main_files = 0
        shard_numbers: list[int] = []
        for entry in files:
            if entry.get("role") not in roles:
                fail(f"model {identifier} has invalid file role")
            relative = Path(str(entry.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                fail(f"model {identifier} contains unsafe relative path {relative}")
            if not isinstance(entry.get("bytes"), int) or entry["bytes"] <= 0:
                fail(f"model {identifier} has invalid expected byte count")
            digest = entry.get("sha256", "")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                fail(f"model {identifier} has invalid SHA-256 for {relative}")
            if entry["role"] == "main":
                main_files += 1
                if "shard" in entry:
                    shard_numbers.append(entry["shard"])
        if shard_numbers and shard_numbers != list(range(1, main_files + 1)):
            fail(f"model {identifier} has incomplete or misordered shards")
        if main_files < 1:
            fail(f"model {identifier} has no main catalog file")
        download = model.get("download")
        if download is not None:
            if (
                not isinstance(download, dict)
                or download.get("provider") != "huggingface"
                or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", str(download.get("repository", "")))
                or not re.fullmatch(r"[0-9a-f]{40}", str(download.get("revision", "")))
                or not isinstance(download.get("allow_patterns"), list)
                or not download["allow_patterns"]
            ):
                fail(f"model {identifier} has invalid pinned download metadata")
            repository_path = Path(download["repository"])
            catalog_paths = [Path(entry["path"]) for entry in files]
            patterns = download["allow_patterns"]
            if (
                model.get("repository") != download["repository"]
                or model.get("revision") != download["revision"]
                or any(path.parent != repository_path for path in catalog_paths)
                or any(not isinstance(pattern, str) or Path(pattern).name != pattern for pattern in patterns)
                or set(patterns) != {path.name for path in catalog_paths}
            ):
                fail(f"model {identifier} download allowlist does not exactly match its catalog files")
        expected_gguf = model.get("gguf_expectations")
        if expected_gguf is not None:
            if (
                model_format != "gguf"
                or not isinstance(expected_gguf, dict)
                or not isinstance(expected_gguf.get("metadata"), dict)
                or not isinstance(expected_gguf.get("array_lengths"), dict)
                or not isinstance(expected_gguf.get("tensor_type_counts"), dict)
                or not isinstance(expected_gguf.get("required_tensors"), list)
                or not isinstance(expected_gguf.get("forbidden_tensor_terms"), list)
            ):
                fail(f"model {identifier} has invalid GGUF semantic expectations")
    for identifier, profile in profiles.items():
        model_id = profile.get("model")
        engine = profile.get("engine")
        if model_id not in models:
            fail(f"profile {identifier} references unknown model {model_id}")
        if engine not in models[model_id]["engines"]:
            fail(f"profile {identifier} uses disallowed engine {engine}")
        context = profile.get("context")
        if not isinstance(context, int) or (engine == "speech" and context != 0) or (engine != "speech" and context < 4096):
            fail(f"profile {identifier} has invalid context")
        features = set(profile.get("features", []))
        settings = profile.get("settings", {})
        if "thinking" in profile and not isinstance(profile["thinking"], bool):
            fail(f"profile {identifier} has an invalid thinking policy")
        if settings.get("load_mode", "mmap") not in {"none", "mmap", "dio"}:
            fail(f"profile {identifier} has invalid load mode")
        if "mtp" in features and ("vision" in features or settings.get("parallel") != 1):
            fail(f"profile {identifier} violates MTP vision/parallel constraints")
        if "vision" in features and not any(item["role"] == "mmproj" for item in models[model_id]["files"]):
            fail(f"profile {identifier} lacks a cataloged projector")
        if "dspark" in features and not any(item["role"] == "dspark" for item in models[model_id]["files"]):
            fail(f"profile {identifier} lacks a cataloged DSpark companion")
        if engine == "rocmfpx":
            integer_settings = {
                "gpu_layers": (1, 999), "parallel": (1, 1), "batch": (1, 8192),
                "ubatch": (1, 8192), "threads": (1, 256), "poll": (0, 100),
                "ctx_checkpoints": (0, 1024), "cache_ram_mb": (0, 1024 * 1024),
                "seed": (0, 2**31 - 1),
            }
            if not re.fullmatch(r"Vulkan[0-9]+", str(settings.get("device", ""))):
                fail(f"profile {identifier} has an invalid ROCmFPX device")
            for key, (minimum, maximum) in integer_settings.items():
                value = settings.get(key)
                if not isinstance(value, int) or not minimum <= value <= maximum:
                    fail(f"profile {identifier} has invalid ROCmFPX setting {key}")
            if settings.get("cache_type_k") != "q8_0" or settings.get("cache_type_v") != "turbo4":
                fail(f"profile {identifier} must use the audited asymmetric ROCmFPX KV types")
            if settings.get("flash_attention") is not True:
                fail(f"profile {identifier} must explicitly enable ROCmFPX Flash Attention")
            if settings.get("temperature") != 0:
                fail(f"profile {identifier} must default to deterministic ROCmFPX sampling")
            speculation_settings = {
                "spec_draft_n_max", "spec_draft_p_min", "spec_mtp_strict_qwen",
            }
            if "mtp" in features:
                if (
                    not isinstance(settings.get("spec_draft_n_max"), int)
                    or not 1 <= settings["spec_draft_n_max"] <= 16
                    or isinstance(settings.get("spec_draft_p_min"), bool)
                    or not isinstance(settings.get("spec_draft_p_min"), (int, float))
                    or not 0 <= settings["spec_draft_p_min"] <= 1
                    or settings.get("spec_mtp_strict_qwen") is not True
                ):
                    fail(f"profile {identifier} must use bounded strict Qwen ROCmFPX MTP")
            elif set(settings).intersection(speculation_settings):
                fail(f"baseline profile {identifier} cannot contain ROCmFPX speculation settings")
            if set(settings) - (
                set(integer_settings)
                | {"device", "cache_type_k", "cache_type_v", "flash_attention", "temperature"}
                | speculation_settings
            ):
                fail(f"profile {identifier} contains an unsupported ROCmFPX setting")
        template_variant = profile.get("chat_template")
        if template_variant is not None:
            matches = [
                item for item in models[model_id]["files"]
                if item["role"] == "chat_template" and item.get("variant") == template_variant
            ]
            if len(matches) != 1:
                fail(f"profile {identifier} requires exactly one {template_variant!r} chat template")


def load_presets(config: Config) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(config.path("HALO_AI_PRESET_DIR").glob("*.json")):
        try:
            item = json.loads(read_text(path))
        except json.JSONDecodeError as exc:
            fail(f"invalid JSON in {path}: {exc}")
        identifier = item.get("id")
        if item.get("schema_version") != 1 or not isinstance(identifier, str):
            fail(f"invalid preset in {path}")
        if identifier in result:
            fail(f"duplicate preset {identifier}")
        result[identifier] = item
    return result


def model_paths(config: Config, model: dict[str, Any], roles: set[str] | None = None) -> list[tuple[dict[str, Any], Path]]:
    root = config.path("HALO_AI_MODELS_ROOT").resolve()
    result = []
    for entry in model["files"]:
        if roles is not None and entry["role"] not in roles:
            continue
        path = root / entry["path"]
        try:
            path.resolve().relative_to(root)
        except ValueError:
            fail(f"catalog path escapes model root: {entry['path']}")
        result.append((entry, path))
    return result


def gguf_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:4] != b"GGUF":
        return {"valid": False, "magic": header[:4] == b"GGUF"}
    version, tensors, metadata = struct.unpack("<IQQ", header[4:])
    valid = version in {2, 3} and tensors < 10_000_000 and 0 < metadata < 1_000_000
    return {"valid": valid, "magic": True, "version": version, "tensors": tensors, "metadata": metadata}


def gguf_inventory(path: Path) -> dict[str, Any]:
    """Parse bounded GGUF metadata and the tensor descriptor table without loading weights."""
    fixed_types: dict[int, tuple[str, int]] = {
        0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2),
        4: ("I", 4), 5: ("i", 4), 6: ("f", 4), 7: ("?", 1),
        10: ("Q", 8), 11: ("q", 8), 12: ("d", 8),
    }
    file_size = path.stat(follow_symlinks=False).st_size
    with path.open("rb") as stream:
        def exact(length: int) -> bytes:
            if length < 0 or length > file_size:
                raise ValueError("GGUF field length is outside the file bounds")
            data = stream.read(length)
            if len(data) != length:
                raise ValueError("truncated GGUF metadata or tensor table")
            return data

        def uint32() -> int:
            return struct.unpack("<I", exact(4))[0]

        def uint64() -> int:
            return struct.unpack("<Q", exact(8))[0]

        def string() -> str:
            length = uint64()
            if length > 64 * 1024 * 1024:
                raise ValueError("GGUF string exceeds the inspection limit")
            return exact(length).decode("utf-8", "strict")

        def value(value_type: int) -> Any:
            if value_type in fixed_types:
                fmt, length = fixed_types[value_type]
                return struct.unpack(f"<{fmt}", exact(length))[0]
            if value_type == 8:
                return string()
            if value_type == 9:
                subtype = uint32()
                count = uint64()
                if count > 10_000_000:
                    raise ValueError("GGUF array exceeds the inspection limit")
                if subtype in fixed_types:
                    stream.seek(fixed_types[subtype][1] * count, os.SEEK_CUR)
                else:
                    for _ in range(count):
                        value(subtype)
                return {"subtype": subtype, "length": count}
            raise ValueError(f"unsupported GGUF metadata value type {value_type}")

        if exact(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = uint32()
        tensor_count = uint64()
        metadata_count = uint64()
        if version not in {2, 3} or tensor_count > 10_000_000 or metadata_count > 1_000_000:
            raise ValueError("invalid GGUF header counts")
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = string()
            metadata[key] = value(uint32())
        tensor_names: list[str] = []
        tensor_type_counts: dict[str, int] = {}
        for _ in range(tensor_count):
            name = string()
            dimensions = uint32()
            if dimensions > 16:
                raise ValueError("GGUF tensor has too many dimensions")
            for _dimension in range(dimensions):
                uint64()
            tensor_type = str(uint32())
            uint64()  # byte offset relative to the aligned tensor-data section
            tensor_names.append(name)
            tensor_type_counts[tensor_type] = tensor_type_counts.get(tensor_type, 0) + 1
    return {
        "version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "metadata": metadata,
        "tensor_type_counts": tensor_type_counts,
        "tensor_names": tensor_names,
    }


def verify_gguf_expectations(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    inventory = gguf_inventory(path)
    checks: dict[str, bool] = {
        "version": inventory["version"] == expected.get("version"),
        "tensor_count": inventory["tensor_count"] == expected.get("tensor_count"),
        "metadata_count": inventory["metadata_count"] == expected.get("metadata_count"),
        "tensor_type_counts": inventory["tensor_type_counts"] == expected.get("tensor_type_counts"),
    }
    observed_metadata: dict[str, Any] = {}
    for key, wanted in expected.get("metadata", {}).items():
        observed_metadata[key] = inventory["metadata"].get(key)
        checks[f"metadata:{key}"] = observed_metadata[key] == wanted
    observed_arrays: dict[str, Any] = {}
    for key, wanted in expected.get("array_lengths", {}).items():
        item = inventory["metadata"].get(key)
        observed_arrays[key] = item.get("length") if isinstance(item, dict) else None
        checks[f"array_length:{key}"] = observed_arrays[key] == wanted
    names = set(inventory["tensor_names"])
    missing = sorted(set(expected.get("required_tensors", [])) - names)
    forbidden = sorted({
        term for term in expected.get("forbidden_tensor_terms", [])
        if any(term.lower() in name.lower() for name in names)
    })
    checks["required_tensors"] = not missing
    checks["forbidden_tensor_terms"] = not forbidden
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "observed_metadata": observed_metadata,
        "observed_array_lengths": observed_arrays,
        "tensor_type_counts": inventory["tensor_type_counts"],
        "missing_tensors": missing,
        "forbidden_tensor_terms_found": forbidden,
    }


def verify_model(config: Config, model: dict[str, Any], full: bool = False) -> dict[str, Any]:
    files = []
    valid = True
    for entry, path in model_paths(config, model):
        record: dict[str, Any] = {"path": str(path), "role": entry["role"], "expected_bytes": entry["bytes"]}
        try:
            info = path.stat(follow_symlinks=False)
            record["bytes"] = info.st_size
            record["mode"] = stat.filemode(info.st_mode)
            if not stat.S_ISREG(info.st_mode) or info.st_size != entry["bytes"]:
                valid = False
            if entry["role"] == "chat_template":
                template = path.read_text(encoding="utf-8")
                template_valid = (
                    path.suffix == ".jinja"
                    and "messages" in template
                    and "add_generation_prompt" in template
                    and "<|im_start|>" in template
                )
                record["jinja_sanity"] = template_valid
                valid = valid and template_valid
            elif model.get("format", "gguf") == "transformers":
                if path.suffix == ".json":
                    try:
                        parsed = json.loads(path.read_text(encoding="utf-8"))
                        format_valid = isinstance(parsed, dict)
                    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                        format_valid = False
                    record["json_sanity"] = format_valid
                elif path.suffix == ".safetensors":
                    with path.open("rb") as stream:
                        raw_length = stream.read(8)
                    header_length = int.from_bytes(raw_length, "little") if len(raw_length) == 8 else 0
                    format_valid = 1 < header_length < info.st_size - 8
                    record["safetensors_sanity"] = format_valid
                else:
                    format_valid = path.suffix == ".model"
                    record["processor_sanity"] = format_valid
                valid = valid and format_valid
            else:
                header = gguf_header(path)
                record["gguf_magic"] = header["magic"]
                record["gguf_header"] = header
                valid = valid and header["valid"]
                expected = model.get("gguf_expectations") if entry["role"] == "main" else None
                if expected:
                    semantics = verify_gguf_expectations(path, expected)
                    record["gguf_semantics"] = semantics
                    valid = valid and semantics["valid"]
            if full:
                digest = hashlib.sha256()
                with path.open("rb", buffering=4 * 1024 * 1024) as stream:
                    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        digest.update(chunk)
                record["sha256"] = digest.hexdigest()
                record["digest_valid"] = record["sha256"] == entry["sha256"]
                valid = valid and record["digest_valid"]
        except (OSError, UnicodeDecodeError, ValueError, struct.error) as exc:
            record["error"] = str(exc)
            valid = False
        files.append(record)
    return {"model": model["id"], "valid": valid, "full": full, "checked_at": utc_now(), "files": files}


def record_full_verifications(config: Config, results: Iterable[dict[str, Any]]) -> None:
    """Merge full model evidence instead of discarding prior verified entries."""
    path = config.path("HALO_AI_INVENTORY_FILE")
    existing: list[dict[str, Any]] = []
    if path.exists():
        try:
            document = json.loads(read_text(path))
        except json.JSONDecodeError:
            fail(f"model inventory is corrupt: {path}")
        if not isinstance(document, dict) or not isinstance(document.get("results"), list):
            fail(f"model inventory has an invalid schema: {path}")
        existing = [item for item in document["results"] if isinstance(item, dict)]
    updates = {item["model"]: item for item in results if item.get("full") is True}
    merged = [item for item in existing if item.get("model") not in updates]
    merged.extend(updates.values())
    merged.sort(key=lambda item: str(item.get("model", "")))
    atomic_json(path, {"schema_version": 1, "results": merged})


def scan_models(config: Config) -> list[dict[str, Any]]:
    root = config.path("HALO_AI_MODELS_ROOT")
    if not root.is_dir():
        fail(f"model root is unavailable: {root}")
    root_device = root.stat().st_dev
    found: list[dict[str, Any]] = []
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories[:] = [
            name for name in directories
            if not name.startswith(".") and not (current_path / name).is_symlink()
            and (current_path / name).stat().st_dev == root_device
        ]
        for name in files:
            if name.startswith(".") or name.startswith("._") or not name.lower().endswith(".gguf"):
                continue
            path = current_path / name
            try:
                info = path.stat(follow_symlinks=False)
                if info.st_dev != root_device or not stat.S_ISREG(info.st_mode):
                    continue
                header = gguf_header(path)
                role = "main"
                lowered = name.lower()
                if lowered.startswith("mmproj"):
                    role = "mmproj"
                elif lowered.startswith("dspark"):
                    role = "dspark"
                found.append({
                    "path": str(path.relative_to(root)), "bytes": info.st_size,
                    "role": role, "gguf_magic": header["magic"], "gguf_header": header,
                    "mode": stat.filemode(info.st_mode), "uid": info.st_uid, "gid": info.st_gid,
                })
            except OSError as exc:
                found.append({"path": str(path), "error": str(exc)})
    return found


def read_int(path: Path) -> int:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return 0


def meminfo_bytes(field: str) -> int:
    prefix = f"{field}:"
    for line in read_text(Path("/proc/meminfo")).splitlines():
        if line.startswith(prefix):
            return int(line.split()[1]) * 1024
    return 0


def hardware_snapshot(config: Config) -> dict[str, Any]:
    selected: Path | None = None
    for candidate in sorted(Path("/sys/class/drm").glob("card*/device")):
        if (candidate / "mem_info_vram_total").is_file() and (candidate / "mem_info_gtt_total").is_file():
            selected = candidate
            break
    if selected is None:
        fail("no amdgpu DRM device exposes VRAM/GTT memory information")
    gtt_total = read_int(selected / "mem_info_gtt_total")
    snapshot = {
        "HALO_AI_DRM_DEVICE": str(selected),
        "HALO_AI_GTT_APERTURE_BYTES": gtt_total,
        "HALO_AI_GTT_APERTURE_GIB": gtt_total // (1024**3),
        "HALO_AI_GTT_USED_BYTES": read_int(selected / "mem_info_gtt_used"),
        "HALO_AI_VRAM_TOTAL_BYTES": read_int(selected / "mem_info_vram_total"),
        "HALO_AI_VRAM_USED_BYTES": read_int(selected / "mem_info_vram_used"),
        "HALO_AI_CPU_MEM_TOTAL_BYTES": meminfo_bytes("MemTotal"),
        "HALO_AI_CPU_MEM_AVAILABLE_BYTES": meminfo_bytes("MemAvailable"),
        "HALO_AI_GTT_TARGET_GIB": config.get("HALO_AI_GTT_TARGET_GIB"),
        "HALO_AI_GTT_PENDING_GIB": "",
    }
    pending = config.path("HALO_AI_STATE_DIR") / "pending-trial.env"
    if pending.exists():
        snapshot["HALO_AI_GTT_PENDING_GIB"] = parse_env_file(pending).get("HALO_AI_GTT_PENDING_GIB", "")
    runtime = runtime_dir() / "hardware.env"
    lines = [f"{key}={shlex.quote(str(value))}" for key, value in snapshot.items()]
    atomic_write(runtime, "\n".join(lines) + "\n")
    return snapshot


def runtime_dir() -> Path:
    base = os.environ.get("XDG_RUNTIME_DIR")
    if not base:
        candidate = Path(f"/run/user/{os.getuid()}")
        if candidate.is_dir():
            base = str(candidate)
    if not base:
        fail("XDG_RUNTIME_DIR is unavailable")
    path = Path(base) / "halo-ai"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.stat().st_uid != os.getuid():
        fail(f"runtime directory is not owned by current user: {path}")
    return path


def run(
    command: list[str], *, check: bool = True, capture: bool = False,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command, check=check, text=True, capture_output=capture, input=input_text,
        )
    except FileNotFoundError:
        fail(f"required command not found: {command[0]}")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        fail(f"command failed ({shlex.join(command)}): {detail or f'exit {exc.returncode}'}")


def podman(
    arguments: list[str], *, check: bool = True, capture: bool = False,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return run(
        ["podman", *arguments], check=check, capture=capture, input_text=input_text,
    )


def container_name(engine: str) -> str:
    if engine not in ENGINE_CONTAINERS:
        fail(f"unsupported engine: {engine}")
    return ENGINE_CONTAINERS[engine]


def image_for(config: Config, engine: str) -> str:
    key = ENGINE_IMAGE_KEYS.get(engine)
    if key is None:
        fail(f"unsupported engine: {engine}")
    image = config.get(key)
    if not image:
        fail(f"{key} is empty; select and test a current image before using {engine}")
    return image


def port_for(config: Config, engine: str) -> int:
    key = ENGINE_PORT_KEYS.get(engine)
    if key is None:
        fail(f"unsupported engine: {engine}")
    return config.integer(key, 1, 65535)


def required_roles(profile: dict[str, Any]) -> set[str]:
    roles = {"main"}
    features = set(profile.get("features", []))
    if "vision" in features:
        roles.add("mmproj")
    if "dspark" in features:
        roles.add("dspark")
    if profile.get("chat_template"):
        roles.add("chat_template")
    return roles


def selected_chat_template(model: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any] | None:
    variant = profile.get("chat_template")
    if not variant:
        return None
    matches = [
        entry for entry in model["files"]
        if entry["role"] == "chat_template" and entry.get("variant") == variant
    ]
    if len(matches) != 1:
        fail(f"profile {profile['id']} requires exactly one {variant!r} chat template")
    return matches[0]


def container_template_path(model: dict[str, Any], profile: dict[str, Any]) -> str | None:
    entry = selected_chat_template(model, profile)
    return None if entry is None else f"/models/templates/{Path(entry['path']).name}"


def lemonade_runtime_mounts(config: Config, catalog: Catalog) -> dict[str, Path]:
    mounts: dict[str, Path] = {}
    for candidate in catalog.models.values():
        if "lemonade" not in candidate["engines"]:
            continue
        for entry, path in model_paths(config, candidate, {"main", "chat_template"}):
            if entry["role"] == "chat_template":
                destination = f"/models/templates/{path.name}"
            else:
                # This folder is deliberately text-only. Lemonade associates a
                # projector with every model identity in a folder, so mixing it
                # here would implicitly turn text and MTP profiles into vision.
                destination = f"/models/extra/{candidate['id']}/{path.name}"
            prior = mounts.get(destination)
            if prior is not None and prior != path:
                fail(f"container mount collision for {destination}: {prior} and {path}")
            mounts[destination] = path
        if any(
            profile["engine"] == "lemonade"
            and profile["model"] == candidate["id"]
            and "vision" in profile.get("features", [])
            for profile in catalog.profiles.values()
        ):
            for entry, path in model_paths(config, candidate, {"main", "mmproj"}):
                destination = f"/models/extra/{candidate['id']}-vision/{path.name}"
                prior = mounts.get(destination)
                if prior is not None and prior != path:
                    fail(f"container mount collision for {destination}: {prior} and {path}")
                mounts[destination] = path
    return mounts


def lemonade_runtime_spec(config: Config, catalog: Catalog) -> str:
    document = {
        "schema": 1,
        "image": image_for(config, "lemonade"),
        "port": port_for(config, "lemonade"),
        "devices": ["/dev/kfd", "/dev/dri"],
        "mounts": sorted((destination, str(path)) for destination, path in lemonade_runtime_mounts(config, catalog).items()),
        "volumes": ["halo-lemonade-huggingface", "halo-lemonade-llama", "halo-lemonade-config"],
        "configuration": {
            "auto_check_model_updates": False,
            "enable_dgpu_gtt": True,
            "extra_models_dir": "/models/extra",
            "llamacpp.backend": "rocm",
            "llamacpp.rocm_bin": config.get("LEMONADE_LLAMACPP_ROCM_BIN"),
            "max_loaded_models": 1,
            "no_broadcast": True,
            "rocm_channel": "stable",
        },
    }
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()[:16]


def llama_server_common_arguments(
    executable: str, profile: dict[str, Any], model_path: str, port: int,
) -> list[str]:
    settings = profile["settings"]
    return [
        executable, "-m", model_path,
        "--host", "0.0.0.0", "--port", str(port), "-c", str(profile["context"]),
        "-np", str(settings["parallel"]), "--batch-size", str(settings["batch"]),
        "--ubatch-size", str(settings["ubatch"]),
        "--flash-attn", "on" if settings.get("flash_attention", True) else "off",
        "--metrics",
    ]


def render_container(config: Config, catalog: Catalog, profile: dict[str, Any]) -> list[str]:
    engine = profile["engine"]
    model = catalog.models[profile["model"]]
    name = container_name(engine)
    port = port_for(config, engine)
    command = [
        "podman", "run", "-d", "--name", name,
        "--label", MANAGED_LABEL,
        "--label", f"local.halo-ai.engine={engine}",
        "--label", f"local.halo-ai.profile={profile['id']}",
        "-p", f"127.0.0.1:{port}:{port}",
    ]
    if engine == "rocmfpx":
        command.extend(["--device=/dev/dri", "--group-add", "keep-groups"])
    else:
        command.extend(["--device=/dev/kfd", "--device=/dev/dri"])
    selected = model_paths(config, model, required_roles(profile))
    if engine == "lemonade":
        command.extend(["--label", f"local.halo-ai.runtime-spec={lemonade_runtime_spec(config, catalog)}"])
        command.extend([
            "-v", "halo-lemonade-huggingface:/opt/lemonade/.cache/huggingface:U",
            "-v", "halo-lemonade-llama:/opt/lemonade/llama:U",
            "-v", "halo-lemonade-config:/opt/lemonade/.cache/lemonade:U",
        ])
        # The long-lived Lemonade service carries exactly the cataloged Qwen text
        # mains and hash-pinned templates. This permits unload/load switching
        # without recreating it while keeping every host mount read-only.
        for destination, path in lemonade_runtime_mounts(config, catalog).items():
            command.extend(["--mount", f"type=bind,src={path},dst={destination},ro"])
        command.append(image_for(config, engine))
    elif engine == "ds4":
        command.extend(["--group-add", "video", "--group-add", "render", "--ipc=host", "--cap-add=SYS_PTRACE", "--security-opt", "seccomp=unconfined"])
        main = next(path for entry, path in selected if entry["role"] == "main")
        command.extend(["--mount", f"type=bind,src={main},dst=/models/model.gguf,ro"])
        use_kv_cache = "kv-cache" in profile.get("features", []) or config.boolean("DS4_KV_CACHE_ENABLED")
        if use_kv_cache:
            command.extend(["--mount", f"type=bind,src={config.path('DS4_KV_CACHE_DIR')},dst=/var/cache/ds4-kv"])
        command.extend([
            image_for(config, engine), "ds4-server", "-m", "/models/model.gguf",
            "--rocm", "--host", "0.0.0.0", "--port", str(port),
            "--ctx", str(profile["context"]),
        ])
        chunk = profile["settings"].get("prefill_chunk")
        if chunk:
            command.extend(["--prefill-chunk", str(chunk)])
        if use_kv_cache:
            cache_mb = str(profile["settings"].get("kv_cache_mb", config.get("DS4_KV_CACHE_MB")))
            command.extend([
                "--kv-disk-dir", "/var/cache/ds4-kv", "--kv-disk-space-mb", cache_mb,
                "--kv-cache-min-tokens", "512", "--kv-cache-cold-max-tokens", "30000",
                "--kv-cache-continued-interval-tokens", "10000",
                "--kv-cache-boundary-trim-tokens", "32",
                "--kv-cache-boundary-align-tokens", "2048",
                "--kv-cache-reject-different-quant",
            ])
    elif engine == "speech":
        command.extend(["--group-add", "keep-groups", "--ipc=host"])
        main = next(path for entry, path in selected if entry["role"] == "main")
        model_directory = main.parent
        command.extend([
            "--mount", f"type=bind,src={model_directory},dst=/models/seamless-m4t-v2-large,ro",
            "-e", "SPEECH_MODEL_PATH=/models/seamless-m4t-v2-large",
            "-e", f"SPEECH_MODEL_REVISION={model['download']['revision']}",
            "-e", f"SPEECH_PORT={port}",
            "-e", "GRADIO_ANALYTICS_ENABLED=False",
            image_for(config, engine),
        ])
    elif engine in {"llamacpp", "rocmfpx"}:
        for index, (entry, path) in enumerate(selected):
            directory = "templates" if entry["role"] == "chat_template" else ""
            destination = f"/models/{directory + '/' if directory else ''}{path.name}"
            command.extend(["--mount", f"type=bind,src={path},dst={destination},ro"])
        main = next(path for entry, path in selected if entry["role"] == "main")
        settings = profile["settings"]
        executable = "/opt/rocmfpx/bin/llama-server" if engine == "rocmfpx" else "llama-server"
        command.append(image_for(config, engine))
        command.extend(llama_server_common_arguments(executable, profile, f"/models/{main.name}", port))
        if engine == "rocmfpx":
            command.extend([
                "--device", settings["device"], "--gpu-layers", str(settings["gpu_layers"]),
                "--ctx-checkpoints", str(settings["ctx_checkpoints"]),
                "--cache-ram", str(settings["cache_ram_mb"]),
                "--threads", str(settings["threads"]), "--poll", str(settings["poll"]),
                "--seed", str(settings["seed"]), "--temp", str(settings["temperature"]),
                "--cache-type-k", settings["cache_type_k"],
                "--cache-type-v", settings["cache_type_v"],
            ])
        else:
            command.extend(["--cache-type-k", settings["kv"], "--cache-type-v", settings["kv"]])
            if settings.get("load_mode"):
                command.extend(["--load-mode", settings["load_mode"]])
        if "vision" in profile.get("features", []):
            projector = next(path for entry, path in selected if entry["role"] == "mmproj")
            command.extend(["--mmproj", f"/models/{projector.name}"])
        if "mtp" in profile.get("features", []):
            command.extend(["--spec-type", "draft-mtp"])
            if engine == "rocmfpx":
                command.extend([
                    "--spec-mtp-strict-qwen",
                    "--spec-draft-n-max", str(settings["spec_draft_n_max"]),
                    "--spec-draft-p-min", str(settings["spec_draft_p_min"]),
                ])
            else:
                command.extend(["--spec-draft-n-max", str(settings["spec_draft_n_max"])])
        if "dspark" in profile.get("features", []):
            companion = next(path for entry, path in selected if entry["role"] == "dspark")
            command.extend(["--spec-type", "draft-dspark", "--model-draft", f"/models/{companion.name}"])
        template_path = container_template_path(model, profile)
        if template_path:
            command.extend(["--chat-template-file", template_path])
        if engine == "rocmfpx" and "mtp" not in profile.get("features", []):
            command.extend(["--spec-type", "none"])
    else:
        fail("vLLM profiles are deferred and cannot be rendered implicitly")
    return command


def lemonade_load_payload(
    profile: dict[str, Any],
    model_name: str = "<resolved-extra-id>",
    chat_template_path: str | None = None,
) -> dict[str, Any]:
    settings = profile["settings"]
    args = [
        "--flash-attn", "on", "--cache-type-k", settings["kv"], "--cache-type-v", settings["kv"],
        "--parallel", str(settings["parallel"]), "--batch-size", str(settings["batch"]),
        "--ubatch-size", str(settings["ubatch"]), "--ctx-checkpoints", "0", "--cache-ram", "0",
        "--cache-reuse", "0",
    ]
    if settings.get("load_mode"):
        args.extend(["--load-mode", settings["load_mode"]])
    if "mtp" in profile.get("features", []):
        args.extend(["--spec-type", "draft-mtp", "--spec-draft-n-max", str(settings["spec_draft_n_max"])])
    else:
        # Lemonade marks these checkpoints as MTP-capable and otherwise enables
        # draft-MTP automatically. Keep baseline profiles genuinely baseline.
        args.extend(["--spec-type", "none"])
    if chat_template_path:
        args.extend(["--chat-template-file", chat_template_path])
    lemonade_managed = {
        "--ctx-size", "--device", "--embedding", "--embeddings", "--jinja", "--metrics",
        "--mmproj", "--mmproj-auto", "--mmproj-offload", "--mmproj-url", "--model",
        "--model-draft", "--no-jinja", "--no-mmproj", "--no-mmproj-auto",
        "--no-mmproj-offload", "--port", "--rerank", "--reranking", "--spec-draft-model",
        "-c", "-dev", "-m", "-md", "-mm", "-mmu",
    }
    conflicts = sorted(lemonade_managed.intersection(args))
    if conflicts:
        fail(f"Lemonade-managed llama.cpp arguments must use first-class fields: {', '.join(conflicts)}")
    return {
        "model_name": model_name, "ctx_size": profile["context"], "llamacpp_backend": "rocm",
        "llamacpp_args": shlex.join(args), "merge_args": False, "save_options": False, "pinned": False,
    }


def managed_containers(all_containers: bool = True) -> list[dict[str, str]]:
    arguments = ["ps", "-a" if all_containers else "", "--filter", f"label={MANAGED_LABEL}", "--format", "json"]
    arguments = [item for item in arguments if item]
    result = podman(arguments, capture=True)
    try:
        return json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        fail("Podman returned invalid container JSON")


def reported_container_name(item: dict[str, Any]) -> str:
    names = item.get("Names") or item.get("Name") or ""
    if isinstance(names, list):
        return str(names[0]) if names else ""
    return str(names)


def ensure_model_mount(config: Config) -> None:
    root = config.path("HALO_AI_MODELS_ROOT")
    if not root.is_dir():
        fail(f"external model root is unavailable: {root}")
    if config.boolean("HALO_AI_MODELS_REQUIRE_MOUNT"):
        result = run(["findmnt", "-n", "-o", "TARGET,UUID", "-T", str(root)], capture=True)
        fields = result.stdout.split()
        if not fields:
            fail(f"cannot resolve model mount for {root}")
        expected = config.get("HALO_AI_MODELS_EXPECT_UUID")
        if expected and (len(fields) < 2 or fields[-1] != expected):
            fail(f"model mount UUID mismatch for {root}")


def assert_port_available(port: int) -> None:
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            fail(f"loopback port {port} is unavailable: {exc}")


def http_json(url: str, *, method: str = "GET", payload: Any = None, timeout: float = 10) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return json.loads(body) if body else None
    except (
        urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
        ConnectionError, json.JSONDecodeError,
    ) as exc:
        fail(f"HTTP request failed for {url}: {exc}")


def container_http_json(
    container: str, url: str, payload: dict[str, Any], *, timeout: int = 600,
) -> Any:
    """POST JSON with curl inside the runtime container, streaming data on stdin."""
    result = podman([
        "exec", "-i", container, "curl", "-fsS", "--max-time", str(timeout),
        "-X", "POST", url, "-H", "Content-Type: application/json",
        "--data-binary", "@-",
    ], capture=True, input_text=json.dumps(payload, separators=(",", ":")))
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        fail(f"container endpoint returned invalid JSON: {url}")


def wait_http(url: str, seconds: int = 120) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2)
    fail(f"service did not become healthy within {seconds}s: {url}")


def service_health_url(config: Config, engine: str) -> str:
    port = port_for(config, engine)
    path = "/live" if engine == "lemonade" else "/healthz" if engine == "speech" else "/v1/models"
    return f"http://127.0.0.1:{port}{path}"


def state_path(config: Config, name: str) -> Path:
    return config.path("HALO_AI_STATE_DIR") / name


def boot_id() -> str:
    return read_text(Path("/proc/sys/kernel/random/boot_id")).strip()


def trial_fingerprint(profile: dict[str, Any], model: dict[str, Any]) -> str:
    document = {
        "profile": profile,
        "model": model["id"],
        "files": [(item["path"], item["bytes"]) for item in model["files"]],
    }
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def begin_trial(config: Config, profile: dict[str, Any], model: dict[str, Any], hardware: dict[str, Any]) -> dict[str, Any]:
    trial = {
        "schema_version": 1, "started_at": utc_now(), "boot_id": boot_id(), "status": "starting",
        "profile": profile["id"], "engine": profile["engine"], "model": model["id"],
        "fingerprint": trial_fingerprint(profile, model), "context": profile["context"],
        "settings": profile["settings"], "features": profile.get("features", []), "hardware": hardware,
    }
    atomic_json(state_path(config, "active-trial.json"), trial)
    return trial


def container_oom_kill_count(container: str) -> int | None:
    result = podman(["inspect", container, "--format", "{{.State.Pid}}"], check=False, capture=True)
    if result.returncode != 0 or not result.stdout.strip().isdigit():
        return None
    pid = result.stdout.strip()
    try:
        cgroup = next(
            line.split(":", 2)[2]
            for line in read_text(Path(f"/proc/{pid}/cgroup")).splitlines()
            if line.startswith("0::")
        )
        events = read_text(Path("/sys/fs/cgroup") / cgroup.lstrip("/") / "memory.events")
        values = dict(line.split() for line in events.splitlines())
        return int(values.get("oom_kill", "0"))
    except (OSError, StopIteration, ValueError):
        return None


def lemonade_backend_versions(container: str) -> dict[str, str]:
    """Report Lemonade's resolved package and the binary serving the model."""
    backends = podman(["exec", container, "lemonade", "backends"], capture=True).stdout
    section_match = re.search(r"(?ms)^llamacpp\s+.*?(?=^\S|\Z)", backends)
    package_match = None if section_match is None else re.search(
        r"(?m)^\s*rocm\s+installed\s+(b[0-9]+)\b", section_match.group(0)
    )
    if package_match is None:
        fail("Lemonade did not report an installed llama.cpp ROCm backend")
    binary = lemonade_backend_binary(container)
    binary_match = re.search(r"/llamacpp/[^/\s]+/llama-(b[0-9]+)/llama-server\b", binary)
    if binary_match is None:
        fail("active llama.cpp path does not contain a build version")
    return {
        "package_version": package_match.group(1),
        "binary_version": binary_match.group(1),
    }


def lemonade_backend_binary(container: str) -> str:
    processes = podman(["top", container, "args"], capture=True).stdout
    binaries = set(re.findall(r"(/opt/lemonade/[^\s]*/llama-server)\b", processes))
    if len(binaries) != 1:
        fail("could not identify exactly one active llama.cpp backend binary")
    return next(iter(binaries))


def assert_lemonade_mtp_model(port: int, model_name: str) -> None:
    models = http_json(f"http://127.0.0.1:{port}/v1/models")
    candidates = models.get("data", models) if isinstance(models, dict) else models
    matches = [
        item for item in candidates or []
        if isinstance(item, dict) and (item.get("id") or item.get("name")) == model_name
    ]
    if len(matches) != 1 or "mtp" not in matches[0].get("labels", []):
        fail(f"Lemonade does not advertise MTP for exact model {model_name}")


def assert_lemonade_mtp_backend(container: str) -> None:
    binary = lemonade_backend_binary(container)
    help_text = podman(["exec", container, binary, "--help"], capture=True).stdout
    required = ("--spec-type", "draft-mtp", "--spec-draft-n-max")
    missing = [flag for flag in required if flag not in help_text]
    if missing:
        fail(f"active llama.cpp backend lacks MTP capability: {', '.join(missing)}")


def finish_trial(config: Config, trial: dict[str, Any], status_value: str, error: str = "") -> None:
    trial.update({"status": status_value, "finished_at": utc_now()})
    if error:
        trial["error"] = error
    atomic_json(state_path(config, "last-trial.json"), trial)
    active = state_path(config, "active-trial.json")
    with contextlib.suppress(FileNotFoundError):
        active.unlink()


def request_trial_stop(config: Config, names: Iterable[str]) -> None:
    """Mark an in-flight start as an operator stop, not a workload failure."""
    active = state_path(config, "active-trial.json")
    if not active.exists():
        return
    try:
        trial = json.loads(read_text(active))
    except json.JSONDecodeError:
        fail(f"active trial record is corrupt: {active}")
    if container_name(str(trial.get("engine", ""))) not in set(names):
        return
    atomic_json(
        state_path(config, "stop-request.json"),
        {
            "boot_id": trial.get("boot_id"),
            "fingerprint": trial.get("fingerprint"),
            "profile": trial.get("profile"),
            "started_at": trial.get("started_at"),
            "requested_at": utc_now(),
        },
    )


def consume_trial_stop(config: Config, trial: dict[str, Any]) -> bool:
    request = state_path(config, "stop-request.json")
    if not request.exists():
        return False
    try:
        marker = json.loads(read_text(request))
    except json.JSONDecodeError:
        fail(f"stop request record is corrupt: {request}")
    matched = all(
        marker.get(key) == trial.get(key)
        for key in ("boot_id", "fingerprint", "profile", "started_at")
    )
    if matched:
        request.unlink()
    return matched


def append_history(config: Config, event: dict[str, Any]) -> None:
    history = state_path(config, "oom-history.jsonl")
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def stage_pressure_reduction(config: Config, trial: dict[str, Any]) -> dict[str, str] | None:
    # Speech has no token context, load mode, or speculative feature to reduce.
    # Its failures require runtime/model diagnosis rather than an irrelevant
    # staged LLM profile mutation.
    if trial.get("engine") == "speech":
        return None
    adjustment: dict[str, str] = {
        "HALO_AI_PENDING_PROFILE": str(trial.get("profile", "")),
        "HALO_AI_PENDING_ORIGIN_BOOT_ID": str(trial.get("boot_id", "")),
    }
    features = trial.get("features", [])
    settings = trial.get("settings", {})
    context = int(trial.get("context", 32768))
    if settings.get("load_mode", "mmap") == "mmap":
        adjustment["HALO_AI_PENDING_LOAD_MODE"] = "none"
    elif "dspark" in features:
        adjustment["HALO_AI_PENDING_DISABLE_FEATURE"] = "dspark"
    elif "mtp" in features:
        adjustment["HALO_AI_PENDING_DISABLE_FEATURE"] = "mtp"
    elif int(settings.get("prefill_chunk", 0)) > 1024:
        adjustment["HALO_AI_PENDING_PREFILL_CHUNK"] = "1024"
    elif context > 32768:
        adjustment["HALO_AI_PENDING_CONTEXT"] = str(max(32768, (int(context * 0.8) // 4096) * 4096))
    else:
        return None
    adjustment["HALO_AI_GTT_PENDING_GIB"] = ""
    lines = [f"{key}={value}" for key, value in adjustment.items()]
    atomic_write(state_path(config, "pending-trial.env"), "\n".join(lines) + "\n")
    return adjustment


def refuse_same_boot_retry(config: Config, profile: dict[str, Any], model: dict[str, Any]) -> None:
    last = state_path(config, "last-trial.json")
    if not last.exists():
        return
    try:
        trial = json.loads(read_text(last))
    except json.JSONDecodeError:
        fail(f"last trial record is corrupt: {last}")
    if (
        trial.get("status") == "failed"
        and trial.get("confirmed_oom") is True
        and trial.get("boot_id") == boot_id()
        and trial.get("fingerprint") == trial_fingerprint(profile, model)
    ):
        fail("the same workload already failed in this boot; inspect logs and stage a future-boot adjustment")


def record_start_failure(config: Config, trial: dict[str, Any], container: str, error: str) -> None:
    result = podman(["logs", "--tail", "200", container], check=False, capture=True)
    excerpt = ((result.stdout or "") + (result.stderr or ""))[-12000:]
    combined = f"{error}\n{excerpt}".lower()
    patterns = ("out of memory", "outofmemory", "hiperroroutofmemory", "memory allocation", "cannot allocate memory")
    before = trial.get("cgroup_oom_kill_before")
    after = container_oom_kill_count(container)
    cgroup_confirmed = isinstance(before, int) and isinstance(after, int) and after > before
    confirmed = any(pattern in combined for pattern in patterns) or cgroup_confirmed
    trial["confirmed_oom"] = confirmed
    if confirmed:
        source = trial.get("engine", "unknown")
        adjustment = stage_pressure_reduction(config, trial) if config.get("HALO_AI_GTT_AUTOTUNE") == "stage" else None
        append_history(
            config,
            {
                **trial,
                "classified_at": utc_now(),
                "classification": "confirmed_oom",
                "confirmed_oom": True,
                "cgroup_oom_kill_before": before,
                "cgroup_oom_kill_after": after,
                "oom_source": source,
                "log_excerpt": excerpt,
                "adjustment": adjustment,
            },
        )


def apply_pending_trial(config: Config, profile: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    pending_path = state_path(config, "pending-trial.env")
    if not pending_path.exists():
        return profile, None
    pending = parse_env_file(pending_path)
    if pending.get("HALO_AI_PENDING_PROFILE") != profile["id"]:
        return profile, None
    if pending.get("HALO_AI_PENDING_ORIGIN_BOOT_ID") == boot_id():
        fail("a pending adjustment cannot be retried in its originating boot")
    adjusted = copy.deepcopy(profile)
    if pending.get("HALO_AI_PENDING_PREFILL_CHUNK"):
        adjusted["settings"]["prefill_chunk"] = int(pending["HALO_AI_PENDING_PREFILL_CHUNK"])
    if pending.get("HALO_AI_PENDING_LOAD_MODE"):
        adjusted["settings"]["load_mode"] = pending["HALO_AI_PENDING_LOAD_MODE"]
    if pending.get("HALO_AI_PENDING_CONTEXT"):
        adjusted["context"] = int(pending["HALO_AI_PENDING_CONTEXT"])
    disabled = pending.get("HALO_AI_PENDING_DISABLE_FEATURE")
    if disabled:
        adjusted["features"] = [item for item in adjusted.get("features", []) if item != disabled]
    return adjusted, pending_path


def detect_prior_lockup(config: Config) -> None:
    active = state_path(config, "active-trial.json")
    if not active.exists():
        # A completed stop can outlive the start process it canceled. The
        # marker has served its purpose once no active trial remains.
        with contextlib.suppress(FileNotFoundError):
            state_path(config, "stop-request.json").unlink()
        return
    try:
        trial = json.loads(read_text(active))
    except json.JSONDecodeError:
        fail(f"active trial record is corrupt: {active}")
    if trial.get("boot_id") == boot_id():
        fail(f"profile {trial.get('profile')} already has an unfinished trial in this boot; inspect logs and stop/discard it")
    event = dict(trial)
    event.update({"classified_at": utc_now(), "classification": "suspected_lockup", "confirmed_oom": False})
    event["adjustment"] = stage_pressure_reduction(config, trial) if config.get("HALO_AI_GTT_AUTOTUNE") == "stage" else None
    append_history(config, event)
    archive = state_path(config, f"suspected-lockup-{int(time.time())}.json")
    os.replace(active, archive)
    fail(f"unfinished prior-boot trial classified as suspected_lockup; review {archive} and the OOM history before starting another profile")


def command_doctor(config: Config, catalog: Catalog, _args: argparse.Namespace) -> int:
    snapshot = hardware_snapshot(config)
    checks: list[tuple[str, str, bool]] = []
    checks.append(("GPU", snapshot["HALO_AI_DRM_DEVICE"], snapshot["HALO_AI_VRAM_TOTAL_BYTES"] > 0))
    checks.append(("GTT aperture", f"{snapshot['HALO_AI_GTT_APERTURE_GIB']} GiB", snapshot["HALO_AI_GTT_APERTURE_GIB"] >= 110))
    cpu_total = snapshot["HALO_AI_CPU_MEM_TOTAL_BYTES"]
    cpu_available = snapshot["HALO_AI_CPU_MEM_AVAILABLE_BYTES"]
    reserve = config.integer("HALO_AI_OS_RESERVE_GIB", 4, 64) * 1024**3
    checks.append((
        "GTT backing topology",
        f"{cpu_total / 1024**3:.1f} GiB CPU-visible for {snapshot['HALO_AI_GTT_APERTURE_GIB']} GiB aperture",
        cpu_total >= snapshot["HALO_AI_GTT_APERTURE_BYTES"] + reserve,
    ))
    for device in (Path("/dev/kfd"), Path("/dev/dri")):
        checks.append((str(device), "present" if device.exists() else "missing", device.exists()))
    checks.append(("Podman", shutil.which("podman") or "missing", shutil.which("podman") is not None))
    root = config.path("HALO_AI_MODELS_ROOT")
    checks.append(("Model root", str(root), root.is_dir()))
    if shutil.which("podman"):
        info = podman(["info", "--format", "json"], check=False, capture=True)
        rootless = False
        if info.returncode == 0:
            with contextlib.suppress(json.JSONDecodeError):
                document = json.loads(info.stdout)
                rootless = bool(document.get("host", {}).get("security", {}).get("rootless"))
        checks.append(("Podman mode", "rootless" if rootless else "not confirmed rootless", rootless))
    checks.append(("CPU MemAvailable", f"{cpu_available / 1024**3:.1f} GiB", cpu_available > reserve))
    pages_limit = read_int(Path("/sys/module/ttm/parameters/pages_limit"))
    checks.append(("TTM pages_limit", str(pages_limit), pages_limit > 0))
    candidates = [int(value) for value in config.get("HALO_AI_GTT_CANDIDATES_GIB").split(",")]
    checks.append(("GTT candidate baseline", str(candidates), snapshot["HALO_AI_GTT_APERTURE_GIB"] in candidates))
    print(f"halo-ai {VERSION} doctor")
    for name, detail, okay in checks:
        print(f"{'PASS' if okay else 'FAIL':4}  {name}: {detail}")
    print(f"INFO  catalog: {len(catalog.models)} models, {len(catalog.profiles)} profiles")
    print(f"INFO  cmdline: {read_text(Path('/proc/cmdline')).strip()}")
    print(f"INFO  NPU: {'active' if Path('/dev/accel/accel0').exists() else 'inactive'}")
    executable: set[str] = set()
    world_readable: set[str] = set()
    for model in catalog.models.values():
        for _entry, path in model_paths(config, model):
            with contextlib.suppress(OSError):
                mode = path.stat(follow_symlinks=False).st_mode
                if mode & 0o111:
                    executable.add(str(path))
                if mode & 0o004:
                    world_readable.add(str(path))
    if executable:
        print(f"WARN  {len(executable)} model files are executable; permissions were not changed")
    if world_readable:
        print(f"WARN  {len(world_readable)} model files are world-readable; permissions were not changed")
    return 0 if all(item[2] for item in checks) else 1


def command_models(config: Config, catalog: Catalog, args: argparse.Namespace) -> int:
    if args.models_action == "list":
        for model in catalog.models.values():
            result = verify_model(config, model, False)
            print(f"{model['id']}\t{'ready' if result['valid'] else 'unavailable'}\t{','.join(model['engines'])}")
        return 0
    if args.models_action == "show":
        model = catalog.models.get(args.model_id)
        if not model:
            fail(f"unknown model: {args.model_id}")
        print(json.dumps({"catalog": model, "verification": verify_model(config, model, False)}, indent=2))
        return 0
    if args.models_action == "scan":
        print(json.dumps(scan_models(config), indent=2))
        return 0
    if args.models_action == "download":
        model = catalog.models.get(args.model_id)
        if not model:
            fail(f"unknown model: {args.model_id}")
        return download_catalog_model(config, model, dry_run=args.dry_run)
    selected = catalog.models.values() if not args.model_id else [catalog.models.get(args.model_id)]
    if None in selected:
        fail(f"unknown model: {args.model_id}")
    results = [verify_model(config, model, args.full) for model in selected]
    print(json.dumps(results, indent=2))
    if args.full:
        record_full_verifications(config, results)
    return 0 if all(item["valid"] for item in results) else 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_matches(entry: dict[str, Any], path: Path) -> bool:
    try:
        info = path.stat(follow_symlinks=False)
        return (
            stat.S_ISREG(info.st_mode)
            and info.st_size == entry["bytes"]
            and sha256_file(path) == entry["sha256"]
        )
    except OSError:
        return False


def approved_download_entries(
    config: Config, model: dict[str, Any], roles: set[str] | None = None,
) -> list[tuple[dict[str, Any], Path]]:
    download = model.get("download")
    if not isinstance(download, dict) or download.get("provider") != "huggingface":
        fail(f"model {model['id']} has no approved pinned downloader")
    selected_roles = None if model.get("format") == "transformers" else roles
    selected = model_paths(config, model, selected_roles)
    allowed = set(download["allow_patterns"])
    if any(path.name not in allowed for _entry, path in selected):
        fail(f"model {model['id']} selected a file outside its download allowlist")
    return selected


def model_acquisition_plan(
    config: Config, model: dict[str, Any], roles: set[str] | None = None,
) -> dict[str, Any]:
    download = model.get("download")
    selected = approved_download_entries(config, model, roles)
    files = []
    additional = 0
    for entry, path in selected:
        verified = file_matches(entry, path)
        if not verified:
            additional += entry["bytes"]
        files.append({
            "role": entry["role"],
            "source": (
                f"https://huggingface.co/{download['repository']}/resolve/"
                f"{download['revision']}/{path.name}"
            ),
            "repository": download["repository"],
            "revision": download["revision"],
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
            "destination": str(path),
            "status": "verified-present" if verified else "download-required",
        })
    return {
        "model": model["id"],
        "files": files,
        "additional_download_bytes": additional,
    }


def download_huggingface_file(
    config: Config, repository: str, revision: str, entry: dict[str, Any], destination: Path,
) -> None:
    if file_matches(entry, destination):
        print(f"Reusing verified artifact: {destination}")
        return
    root = config.path("HALO_AI_MODELS_ROOT").resolve()
    try:
        destination.resolve().relative_to(root)
    except ValueError:
        fail(f"model download path escapes external model root: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    lock_path = destination.with_name(f".{destination.name}.lock")
    url = f"https://huggingface.co/{repository}/resolve/{revision}/{destination.name}"
    try:
        lock_fd = os.open(
            lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600,
        )
    except OSError as exc:
        fail(f"cannot create safe model download lock {lock_path}: {exc}")
    with os.fdopen(lock_fd, "a+b") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if file_matches(entry, destination):
            print(f"Reusing verified artifact: {destination}")
            return
        if partial.exists() and not stat.S_ISREG(partial.stat(follow_symlinks=False).st_mode):
            fail(f"refusing unsafe partial download path: {partial}")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > entry["bytes"]:
            partial.unlink()
            offset = 0
        if offset < entry["bytes"]:
            headers = {"User-Agent": f"halo-ai/{VERSION}"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            try:
                response = urllib.request.urlopen(request, timeout=120)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                fail(f"download failed for {url}: {exc}")
            status_code = getattr(response, "status", response.getcode())
            append = offset > 0 and status_code == 206
            if offset and not append:
                offset = 0
            flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
            flags |= os.O_APPEND if append else os.O_TRUNC
            print(f"Downloading {repository}@{revision}/{destination.name} from byte {offset}")
            try:
                partial_fd = os.open(partial, flags, 0o640)
            except OSError as exc:
                response.close()
                fail(f"cannot create safe partial download {partial}: {exc}")
            with response, os.fdopen(partial_fd, "ab" if append else "wb") as stream:
                os.fchmod(stream.fileno(), 0o640)
                for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        actual_size = partial.stat().st_size if partial.exists() else 0
        actual_sha = sha256_file(partial) if actual_size == entry["bytes"] else ""
        if actual_size != entry["bytes"] or actual_sha != entry["sha256"]:
            with contextlib.suppress(FileNotFoundError):
                partial.unlink()
            fail(
                f"download verification failed for {destination.name}: "
                f"expected {entry['bytes']} bytes/{entry['sha256']}, got {actual_size} bytes/{actual_sha or 'not hashed'}"
            )
        os.replace(partial, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def download_catalog_model(
    config: Config, model: dict[str, Any], *, roles: set[str] | None = None,
    dry_run: bool = False,
) -> int:
    ensure_model_mount(config)
    plan = model_acquisition_plan(config, model, roles)
    print(json.dumps(plan, indent=2))
    if dry_run:
        return 0
    download = model["download"]
    for entry, destination in approved_download_entries(config, model, roles):
        download_huggingface_file(
            config, download["repository"], download["revision"], entry, destination,
        )
    result = verify_model(config, model, True)
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


def profile_availability(config: Config, catalog: Catalog, profile: dict[str, Any]) -> tuple[bool, str]:
    if profile.get("gate"):
        return False, str(profile["gate"])
    key = ENGINE_IMAGE_KEYS.get(profile["engine"])
    if key is None:
        return False, f"unsupported engine: {profile['engine']}"
    if not config.get(key):
        return False, f"{key} is empty"
    result = verify_model(config, catalog.models[profile["model"]], False)
    if not result["valid"]:
        return False, "model verification failed"
    if profile["engine"] == "rocmfpx" and not rocmfpx_image_valid(config.get(key)):
        return False, "pinned ROCmFPX image is not installed or failed provenance checks"
    model = catalog.models[profile["model"]]
    memory_roles = {"weights"} if model.get("format") == "transformers" else {"main"}
    main_bytes = sum(int(entry["bytes"]) for entry in model["files"] if entry["role"] in memory_roles)
    reserve = config.integer("HALO_AI_OS_RESERVE_GIB", 4, 64) * 1024**3
    cpu_total = meminfo_bytes("MemTotal")
    if main_bytes + reserve > cpu_total:
        return False, (
            f"unsafe memory topology: {main_bytes / 1024**3:.1f} GiB weights + "
            f"{reserve / 1024**3:.0f} GiB reserve exceed {cpu_total / 1024**3:.1f} GiB CPU-visible RAM"
        )
    if profile["engine"] == "lemonade":
        for model in catalog.models.values():
            if "lemonade" in model["engines"] and not verify_model(config, model, False)["valid"]:
                return False, f"Lemonade allowlist model unavailable: {model['id']}"
    return True, "ready"


def command_profiles(config: Config, catalog: Catalog, args: argparse.Namespace) -> int:
    if args.profiles_action == "list":
        for profile in catalog.profiles.values():
            ready, reason = profile_availability(config, catalog, profile)
            print(f"{profile['id']}\t{profile['engine']}\t{'ready' if ready else 'disabled'}\t{reason}")
        return 0
    profile = catalog.profiles.get(args.profile_id)
    if not profile:
        fail(f"unknown profile: {args.profile_id}")
    if args.profiles_action == "acquire":
        return command_profile_acquire(config, catalog, profile, dry_run=args.dry_run)
    if args.profiles_action == "show":
        ready, reason = profile_availability(config, catalog, profile)
        output = dict(profile)
        output["availability"] = {"ready": ready, "reason": reason}
        print(json.dumps(output, indent=2))
    else:
        print(shlex.join(render_container(config, catalog, profile)))
        if profile["engine"] == "lemonade":
            model = catalog.models[profile["model"]]
            print(json.dumps(lemonade_load_payload(profile, chat_template_path=container_template_path(model, profile)), indent=2))
    return 0


def profile_acquisition_plan(config: Config, catalog: Catalog, profile: dict[str, Any]) -> dict[str, Any]:
    model = catalog.models[profile["model"]]
    model_plan = model_acquisition_plan(config, model, required_roles(profile))
    engine = profile["engine"]
    image = image_for(config, engine)
    installed = rocmfpx_image_valid(image) if engine == "rocmfpx" else bool(image_identity(image, required=False))
    runtime: dict[str, Any] = {
        "engine": engine,
        "image": image,
        "status": "installed" if installed else "install-required",
        "additional_download_bytes": 0,
    }
    if engine == "rocmfpx":
        runtime.update({
            "q38rocm_commit": ROCMFPX_Q38ROCM_COMMIT,
            "engine_archive": ROCMFPX_ENGINE_URL,
            "engine_archive_bytes": ROCMFPX_ENGINE_BYTES,
            "engine_archive_sha256": ROCMFPX_ENGINE_SHA256,
            "vulkan_base_digest": ROCMFPX_VULKAN_BASE,
            "rocm_base_digest": ROCMFPX_ROCM_BASE,
            "source_provenance": "upstream binary reports build 213 (e87d53e); full source SHA unresolved",
            "additional_download_bytes": 0 if installed else ROCMFPX_ENGINE_BYTES,
        })
    return {
        "schema_version": 1,
        "profile": profile["id"],
        "runtime": runtime,
        "model": model_plan,
        "selected_artifact_classes": (
            ["fp4", "rocmfpx-runtime"]
            + (["in-gguf-mtp"] if "mtp" in profile.get("features", []) else [])
            if engine == "rocmfpx" else ["model", "runtime"]
        ),
        "excluded_artifact_classes": ["fp8", "npu", "vision", "bf16", "reference"],
        "total_additional_download_bytes": (
            model_plan["additional_download_bytes"] + runtime["additional_download_bytes"]
        ),
    }


def command_profile_acquire(
    config: Config, catalog: Catalog, profile: dict[str, Any], *, dry_run: bool,
) -> int:
    plan = profile_acquisition_plan(config, catalog, profile)
    print(json.dumps(plan, indent=2))
    if dry_run:
        return 0
    model = catalog.models[profile["model"]]
    result = download_catalog_model(config, model, roles=required_roles(profile))
    if result != 0:
        return result
    return command_runtime_install(config, [profile["engine"]])


def command_presets(config: Config, args: argparse.Namespace) -> int:
    presets = load_presets(config)
    if args.presets_action == "list":
        for item in presets.values():
            print(f"{item['id']}\t{item.get('description', '')}")
        return 0
    item = presets.get(args.preset_id)
    if not item:
        fail(f"unknown preset: {args.preset_id}")
    print(json.dumps(item if args.presets_action == "show" else item["request"], indent=2))
    return 0


def command_runtime_install(config: Config, engines: Iterable[str]) -> int:
    for engine in engines:
        image = image_for(config, engine)
        if engine == "speech":
            context = PROJECT_ROOT / "lib/halo_ai/speech"
            print(f"Building {engine}: {image} from {context}")
            podman(["build", "--pull=missing", "-t", image, "-f", str(context / "Containerfile"), str(context)])
            ensure_speech_test_audio(config)
        elif engine == "rocmfpx":
            build_rocmfpx_image(image)
        else:
            print(f"Pulling {engine}: {image}")
            podman(["pull", image])
        if engine == "lemonade":
            for name, component in (
                ("halo-lemonade-huggingface", "lemonade-huggingface"),
                ("halo-lemonade-llama", "lemonade-llama"),
                ("halo-lemonade-config", "lemonade-config"),
            ):
                exists = podman(["volume", "exists", name], check=False).returncode == 0
                if not exists:
                    podman(["volume", "create", "--label", MANAGED_LABEL, "--label", f"local.halo-ai.component={component}", name])
        digest = image_identity(image)
        state = config.path("HALO_AI_STATE_DIR")
        state.mkdir(parents=True, exist_ok=True)
        atomic_write(state / f"{engine}-image-digest", digest + "\n")
    return 0


def command_update(config: Config, engines: Iterable[str]) -> int:
    history_path = state_path(config, "image-update-history.jsonl")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    for engine in engines:
        image = image_for(config, engine)
        old = image_identity(image, required=False)
        if engine == "speech":
            context = PROJECT_ROOT / "lib/halo_ai/speech"
            podman(["build", "--pull=always", "-t", image, "-f", str(context / "Containerfile"), str(context)])
            ensure_speech_test_audio(config)
        elif engine == "rocmfpx":
            build_rocmfpx_image(image, force=True)
        else:
            podman(["pull", image])
        new = image_identity(image)
        event = {"updated_at": utc_now(), "engine": engine, "image": image, "old_digest": old, "new_digest": new}
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        atomic_write(state_path(config, f"{engine}-image-digest"), new + "\n")
        print(json.dumps(event))
        if old and old != new:
            print(f"Rollback image reference: {old}")
    print("Images were staged only; active containers were not recreated. Run the profile smoke test before switching.")
    return 0


def image_labels(image: str) -> dict[str, str]:
    result = podman(
        ["image", "inspect", image, "--format", "{{json .Config.Labels}}"],
        check=False, capture=True,
    )
    if result.returncode != 0:
        return {}
    try:
        labels = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    return labels if isinstance(labels, dict) else {}


def rocmfpx_image_valid(image: str) -> bool:
    labels = image_labels(image)
    expected = {
        "local.halo-ai.engine": "rocmfpx",
        "local.halo-ai.engine-archive-sha256": ROCMFPX_ENGINE_SHA256,
        "org.opencontainers.image.revision": ROCMFPX_Q38ROCM_COMMIT,
        "local.halo-ai.vulkan-base": ROCMFPX_VULKAN_BASE,
        "local.halo-ai.rocm-base": ROCMFPX_ROCM_BASE,
    }
    return expected.items() <= labels.items()


def build_rocmfpx_image(image: str, *, force: bool = False) -> None:
    if not force and rocmfpx_image_valid(image):
        print(f"Reusing verified ROCmFPX image: {image}")
        return
    context = PROJECT_ROOT / "lib/halo_ai/rocmfpx"
    containerfile = context / "Containerfile"
    if not containerfile.is_file():
        fail(f"ROCmFPX build recipe is missing: {containerfile}")
    print(f"Building rocmfpx: {image} from immutable base images and engine archive")
    podman([
        "build", "--pull=missing", "--timestamp", "0", "--layers=false",
        "-t", image, "-f", str(containerfile), str(context),
    ])
    if not rocmfpx_image_valid(image):
        fail("built ROCmFPX image failed its immutable label checks")
    version = podman([
        "run", "--rm", "--entrypoint", "/opt/rocmfpx/bin/llama-server",
        image, "--version",
    ], capture=True)
    combined = f"{version.stdout}\n{version.stderr}"
    if "version: 213 (e87d53e)" not in combined:
        fail("built ROCmFPX image did not report the pinned upstream engine build")
    devices = podman([
        "run", "--rm", "--device=/dev/dri", "--group-add", "keep-groups",
        "--entrypoint", "/opt/rocmfpx/bin/llama-server", image, "--list-devices",
    ], capture=True)
    device_output = f"{devices.stdout}\n{devices.stderr}"
    if "Vulkan0" not in device_output or "GFX1151" not in device_output:
        fail("built ROCmFPX image did not expose the required Vulkan0 gfx1151 device")


def rocmfpx_backend_info(
    container: str, image: str, profile: dict[str, Any],
) -> dict[str, Any]:
    version = podman(
        ["exec", container, "/opt/rocmfpx/bin/llama-server", "--version"],
        capture=True,
    )
    combined = f"{version.stdout}\n{version.stderr}"
    if "version: 213 (e87d53e)" not in combined:
        fail("active ROCmFPX backend does not match the pinned engine build")
    processes = podman(["top", container, "args"], capture=True).stdout
    mtp = "mtp" in profile.get("features", [])
    expected_speculation = "draft-mtp" if mtp else "none"
    if (
        "/opt/rocmfpx/bin/llama-server" not in processes
        or not re.search(r"(?:^|\s)--device\s+Vulkan0(?:\s|$)", processes)
        or not re.search(
            rf"(?:^|\s)--spec-type\s+{re.escape(expected_speculation)}(?:\s|$)",
            processes,
        )
        or (mtp and "--spec-mtp-strict-qwen" not in processes)
    ):
        fail("active ROCmFPX process does not match the audited Vulkan0 profile")
    labels = image_labels(image)
    if not rocmfpx_image_valid(image):
        fail("active ROCmFPX image failed provenance checks")
    return {
        "engine_version": "213",
        "reported_source_revision": "e87d53e-unresolved",
        "q38rocm_release_commit": labels["org.opencontainers.image.revision"],
        "engine_archive_sha256": labels["local.halo-ai.engine-archive-sha256"],
        "vulkan_base_digest": labels["local.halo-ai.vulkan-base"],
        "rocm_base_digest": labels["local.halo-ai.rocm-base"],
        "device": "Vulkan0",
        "speculation": expected_speculation,
        "strict_qwen_mtp": mtp,
    }


def image_identity(image: str, *, required: bool = True) -> str:
    result = podman(
        ["image", "inspect", image, "--format", "{{if .Digest}}{{.Digest}}{{else}}{{.Id}}{{end}}"],
        check=required,
        capture=True,
    )
    return result.stdout.strip()


def ensure_speech_test_audio(config: Config) -> Path:
    destination = config.path("SPEECH_TEST_AUDIO_PATH")
    expected_bytes = 566_482
    expected_sha256 = "640d14d5002fb34a9a3cd1d663ea354096de623b687dff75b59ac58943cbcd7f"
    if destination.is_file() and destination.stat().st_size == expected_bytes:
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest == expected_sha256:
            return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".input1.wav.", dir=destination.parent)
    digest = hashlib.sha256()
    copied = 0
    try:
        os.fchmod(fd, 0o640)
        with (
            urllib.request.urlopen(
                "https://developer.amd.com/playbooks/playbook-files/supplemental/"
                "speech2speech-translation/assets/input1.wav",
                timeout=120,
            ) as response,
            os.fdopen(fd, "wb") as stream,
        ):
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if copied != expected_bytes or digest.hexdigest() != expected_sha256:
            fail("AMD speech smoke asset failed its pinned size/SHA-256 check")
        os.replace(temporary, destination)
        return destination
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def exact_lemonade_model(config: Config, catalog: Catalog, profile: dict[str, Any], port: int) -> str:
    models = http_json(f"http://127.0.0.1:{port}/api/v1/models")
    main = next(
        path for entry, path in model_paths(config, catalog.models[profile["model"]], {"main"})
        if entry["role"] == "main"
    )
    candidates = models.get("data", models) if isinstance(models, dict) else models
    matches = []
    for item in candidates or []:
        if not isinstance(item, dict):
            continue
        resolved = item.get("id") or item.get("model_name") or item.get("name")
        checkpoints = item.get("checkpoints") if isinstance(item.get("checkpoints"), dict) else {}
        identity = profile["model"] + ("-vision" if "vision" in profile.get("features", []) else "")
        expected_folder = f"/models/extra/{identity}"
        exact_folder_model = (
            resolved == identity
            and item.get("source") == "extra_models_dir"
            and (checkpoints.get("main") == expected_folder or item.get("checkpoint") == expected_folder)
        )
        # Preserve compatibility with older flattened extra_models_dir layouts,
        # whose API record contains the exact GGUF filename instead.
        exact_file_model = main.name in json.dumps(item)
        if resolved and (exact_folder_model or exact_file_model):
            matches.append(resolved)
    if len(matches) != 1:
        fail(f"expected one exact Lemonade model for {main.name}, found {len(matches)}")
    return matches[0]


def lemonade_backend_running(container: str) -> bool:
    result = podman(["top", container, "comm"], check=False, capture=True)
    return result.returncode == 0 and any(
        line.strip() == "llama-server" for line in result.stdout.splitlines()[1:]
    )


def unload_active_lemonade(config: Config, catalog: Catalog, port: int, container: str) -> None:
    """Unload recorded weights and wait for the backend process to disappear."""
    if not lemonade_backend_running(container):
        return
    active_path = state_path(config, "active-profile.json")
    if not active_path.exists():
        fail("Lemonade has a loaded backend but no active-profile record")
    prior_id = json.loads(read_text(active_path)).get("profile")
    prior = catalog.profiles.get(prior_id)
    if not prior or prior["engine"] != "lemonade":
        fail("active-profile record does not identify the loaded Lemonade model")
    prior_model = exact_lemonade_model(config, catalog, prior, port)
    http_json(
        f"http://127.0.0.1:{port}/v1/unload", method="POST",
        payload={"model_name": prior_model}, timeout=180,
    )
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if not lemonade_backend_running(container):
            return
        time.sleep(1)
    fail("Lemonade backend did not exit within 120 seconds after unload")


def configure_lemonade_runtime(config: Config, container: str, health_url: str) -> None:
    """Apply runtime policy, tolerating one successful backend hot-swap disconnect."""
    arguments = [
        "exec", container, "lemonade", "config", "set",
        "enable_dgpu_gtt=true",
        "extra_models_dir=/models/extra",
        "llamacpp.backend=rocm",
        f"llamacpp.rocm_bin={config.get('LEMONADE_LLAMACPP_ROCM_BIN')}",
        "rocm_channel=stable",
        "max_loaded_models=1",
        "no_broadcast=true",
        "auto_check_model_updates=false",
    ]
    result = podman(arguments, check=False, capture=True)
    if result.returncode == 0:
        return
    # Changing *_bin may replace the backend and restart lemond before the
    # config client receives its response. A healthy service plus a successful
    # idempotent retry proves that this was a completed hot-swap, not a failure.
    wait_http(health_url, 180)
    retry = podman(arguments, check=False, capture=True)
    if retry.returncode != 0:
        detail = (retry.stderr or retry.stdout or result.stderr or result.stdout or "").strip()
        fail(f"could not apply Lemonade runtime configuration: {detail or 'config client failed'}")


def command_start(config: Config, catalog: Catalog, args: argparse.Namespace) -> int:
    profile = catalog.profiles.get(args.profile_id)
    if not profile:
        fail(f"unknown profile: {args.profile_id}")
    profile, pending_path = apply_pending_trial(config, profile)
    ready, reason = profile_availability(config, catalog, profile)
    if not ready:
        fail(f"profile is disabled: {reason}")
    ensure_model_mount(config)
    detect_prior_lockup(config)
    refuse_same_boot_retry(config, profile, catalog.models[profile["model"]])
    hardware = hardware_snapshot(config)
    active = managed_containers(False)
    target_name = container_name(profile["engine"])
    conflicting = [item for item in active if reported_container_name(item) != target_name]
    if conflicting and not args.switch:
        fail("another managed inference runtime is active; stop it or pass --switch")
    if conflicting:
        for item in conflicting:
            name = reported_container_name(item)
            podman(["stop", "--time", "120", name])
    existing = podman(["container", "exists", target_name], check=False).returncode == 0
    reuse_lemonade = False
    if existing:
        labels = podman(["inspect", target_name, "--format", "{{json .Config.Labels}}"], capture=True).stdout
        container_labels = json.loads(labels)
        if container_labels.get("local.halo-ai.managed") != "true":
            fail(f"container name {target_name} exists without the managed label")
        running = podman(["inspect", target_name, "--format", "{{.State.Running}}"], capture=True).stdout.strip() == "true"
        reuse_lemonade = (
            profile["engine"] == "lemonade"
            and running
            and container_labels.get("local.halo-ai.runtime-spec") == lemonade_runtime_spec(config, catalog)
        )
        if not reuse_lemonade:
            podman(["rm", "-f", "--time", "120", target_name])
    if not reuse_lemonade:
        assert_port_available(port_for(config, profile["engine"]))
    trial = begin_trial(config, profile, catalog.models[profile["model"]], hardware)
    try:
        port = port_for(config, profile["engine"])
        health = service_health_url(config, profile["engine"])
        refresh_latest = (
            reuse_lemonade
            and profile["engine"] == "lemonade"
            and config.get("LEMONADE_LLAMACPP_ROCM_BIN") == "latest"
        )
        if refresh_latest:
            # Lemonade resolves the newest channel-compatible backend once per
            # lemond process. Restart on every explicit start when the operator
            # selected `latest`; persistent volumes make this a cached check.
            unload_active_lemonade(config, catalog, port, target_name)
            podman(["restart", "--time", "120", target_name])
            wait_http(health, 600 if profile["engine"] == "speech" else 180)
        elif reuse_lemonade:
            unload_active_lemonade(config, catalog, port, target_name)
        else:
            command = render_container(config, catalog, profile)
            run(command)
            wait_http(health, 600 if profile["engine"] == "speech" else 180)
        trial["cgroup_oom_kill_before"] = container_oom_kill_count(target_name)
        atomic_json(state_path(config, "active-trial.json"), trial)
        if profile["engine"] == "lemonade" and not reuse_lemonade:
            configure_lemonade_runtime(config, target_name, health)
            podman(["restart", "--time", "120", target_name])
            wait_http(health, 180)
        if profile["engine"] == "lemonade":
            match = exact_lemonade_model(config, catalog, profile, port)
            if "mtp" in profile.get("features", []):
                assert_lemonade_mtp_model(port, match)
            model = catalog.models[profile["model"]]
            payload = lemonade_load_payload(
                profile,
                match,
                chat_template_path=container_template_path(model, profile),
            )
            http_json(f"http://127.0.0.1:{port}/v1/load", method="POST", payload=payload, timeout=600)
            if "mtp" in profile.get("features", []):
                assert_lemonade_mtp_backend(target_name)
            trial["backend"] = lemonade_backend_versions(target_name)
        elif profile["engine"] == "rocmfpx":
            trial["backend"] = rocmfpx_backend_info(
                target_name, image_for(config, profile["engine"]), profile,
            )
        trial["container"] = target_name
        trial["image"] = image_for(config, profile["engine"])
        digest_path = state_path(config, f"{profile['engine']}-image-digest")
        if digest_path.exists():
            trial["image_digest"] = read_text(digest_path).strip()
        process_output = podman(["top", target_name, "args"], capture=True).stdout.splitlines()
        trial["process_args"] = process_output[1:]
        trial["hardware_loaded"] = hardware_snapshot(config)
        finish_trial(config, trial, "loaded")
        if pending_path is not None and pending_path.exists():
            os.replace(pending_path, state_path(config, f"applied-pending-{int(time.time())}.env"))
        atomic_json(state_path(config, "active-profile.json"), {"profile": profile["id"], "container": target_name, "started_at": utc_now()})
        print(f"Started {profile['id']} on 127.0.0.1:{port}")
        return 0
    except BaseException as exc:
        if consume_trial_stop(config, trial):
            finish_trial(config, trial, "stopped", "operator requested halo-ai stop")
            eprint("Start canceled by halo-ai stop.")
            return 130
        record_start_failure(config, trial, target_name, str(exc))
        finish_trial(config, trial, "failed", str(exc))
        raise


def stop_names(names: Iterable[str]) -> None:
    for name in names:
        exists = podman(["container", "exists", name], check=False).returncode == 0
        if not exists:
            continue
        labels = json.loads(podman(["inspect", name, "--format", "{{json .Config.Labels}}"], capture=True).stdout)
        if labels.get("local.halo-ai.managed") != "true":
            fail(f"refusing unmanaged container: {name}")
        running = podman(["inspect", name, "--format", "{{.State.Running}}"], capture=True).stdout.strip() == "true"
        if running:
            podman(["stop", "--time", "120", name])


def command_stop(config: Config, catalog: Catalog, args: argparse.Namespace) -> int:
    if not args.target or args.target == "all":
        names = [container_name(engine) for engine in ENGINE_CONTAINERS]
    elif args.target in catalog.profiles:
        names = [container_name(catalog.profiles[args.target]["engine"])]
    else:
        fail(f"unknown profile or target: {args.target}")
    request_trial_stop(config, names)
    stop_names(names)
    active_trial = state_path(config, "active-trial.json")
    if active_trial.exists():
        try:
            trial = json.loads(read_text(active_trial))
        except json.JSONDecodeError:
            fail(f"active trial record is corrupt: {active_trial}")
        if container_name(str(trial.get("engine", ""))) in names:
            finish_trial(config, trial, "stopped", "operator requested halo-ai stop")
    active = state_path(config, "active-profile.json")
    if active.exists():
        try:
            record = json.loads(read_text(active))
        except json.JSONDecodeError:
            record = {}
        if args.target == "all" or record.get("container") in names:
            active.unlink()
    print("Stopped requested managed runtime(s); models, images, caches, and volumes were preserved.")
    return 0


def command_status(config: Config, _catalog: Catalog, _args: argparse.Namespace) -> int:
    snapshot = hardware_snapshot(config)
    containers = managed_containers(True)
    active_path = state_path(config, "active-profile.json")
    active = json.loads(read_text(active_path)) if active_path.exists() else None
    endpoint: dict[str, Any] | None = None
    backend_info: Any = None
    if active and active.get("profile") in _catalog.profiles:
        profile = _catalog.profiles[active["profile"]]
        port = port_for(config, profile["engine"])
        url = service_health_url(config, profile["engine"])
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                endpoint = {"url": url, "healthy": response.status < 500, "status": response.status}
            if profile["engine"] == "lemonade":
                with contextlib.suppress(HaloError):
                    backend_info = http_json(f"http://127.0.0.1:{port}/api/v1/system-info", timeout=2)
            elif profile["engine"] == "rocmfpx":
                try:
                    backend_info = rocmfpx_backend_info(
                        container_name("rocmfpx"), image_for(config, "rocmfpx"), profile,
                    )
                except HaloError as exc:
                    backend_info = {"valid": False, "error": str(exc)}
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            endpoint = {"url": url, "healthy": False, "error": str(exc)}
    digests = {}
    for engine in ENGINE_CONTAINERS:
        path = state_path(config, f"{engine}-image-digest")
        if path.exists():
            digests[engine] = read_text(path).strip()
    pending = state_path(config, "pending-trial.env")
    last_path = state_path(config, "last-trial.json")
    last_trial = json.loads(read_text(last_path)) if last_path.exists() else None
    last_smoke = None
    if isinstance(last_trial, dict) and isinstance(last_trial.get("smoke_tests"), list):
        last_smoke = last_trial["smoke_tests"][-1] if last_trial["smoke_tests"] else None
    print(json.dumps({
        "hardware": snapshot,
        "containers": containers,
        "active_profile": active,
        "endpoint": endpoint,
        "backend_info": backend_info,
        "last_trial_backend": last_trial.get("backend") if isinstance(last_trial, dict) else None,
        "last_smoke": last_smoke,
        "image_digests": digests,
        "pending_trial": parse_env_file(pending) if pending.exists() else None,
    }, indent=2))
    return 0


def command_logs(catalog: Catalog, args: argparse.Namespace) -> int:
    if args.profile_id:
        profile = catalog.profiles.get(args.profile_id)
        if not profile:
            fail(f"unknown profile: {args.profile_id}")
        name = container_name(profile["engine"])
    else:
        active = managed_containers(False)
        if len(active) != 1:
            fail("specify a profile unless exactly one managed container is running")
        name = reported_container_name(active[0])
    command = ["podman", "logs"]
    if args.follow:
        command.append("-f")
    command.append(name)
    return run(command, check=False).returncode


def validate_smoke_response(response: Any, expected: str, thinking: bool) -> None:
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        fail("smoke response did not contain exactly one choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        fail("smoke response choice has no message")
    content = message.get("content")
    if not isinstance(content, str) or content.strip() != expected:
        fail(f"smoke response content mismatch: expected {expected!r}, got {content!r}")
    reasoning = message.get("reasoning_content")
    if not thinking and isinstance(reasoning, str) and reasoning.strip():
        fail("non-thinking smoke response unexpectedly contained reasoning_content")


def validate_lemonade_reasoning_template(container: str, payload: dict[str, Any], thinking: bool) -> None:
    """Verify the active llama.cpp template applied the requested reasoning mode."""
    processes = podman(["top", container, "args"], capture=True).stdout
    ports = set(re.findall(r"/llama-server\b[^\n]*?--port\s+([0-9]+)\b", processes))
    if len(ports) != 1:
        fail("could not identify exactly one active Lemonade llama.cpp backend port")
    template_request: dict[str, Any] = {
        "messages": payload["messages"],
        "add_generation_prompt": True,
    }
    for key in ("chat_template_kwargs", "reasoning_effort"):
        if key in payload:
            template_request[key] = payload[key]
    result = podman([
        "exec", container, "curl", "-fsS", "-X", "POST",
        f"http://127.0.0.1:{next(iter(ports))}/apply-template",
        "-H", "Content-Type: application/json",
        "--data", json.dumps(template_request, separators=(",", ":")),
    ], capture=True)
    try:
        prompt = json.loads(result.stdout)["prompt"]
    except (json.JSONDecodeError, KeyError, TypeError):
        fail("active llama.cpp backend returned an invalid apply-template response")
    disabled_suffix = "<think>\n\n</think>"
    rendered_disabled = isinstance(prompt, str) and prompt.rstrip().endswith(disabled_suffix)
    if thinking and rendered_disabled:
        fail("thinking request rendered the non-thinking chat-template suffix")
    if not thinking and not rendered_disabled:
        fail("non-thinking request did not render the disabled-reasoning suffix")


def merge_request_preset(payload: dict[str, Any], request: dict[str, Any]) -> None:
    template_kwargs = request.get("chat_template_kwargs")
    if (
        isinstance(template_kwargs, dict)
        and template_kwargs.get("enable_thinking") is True
        and "reasoning_effort" not in request
    ):
        payload.pop("reasoning_effort", None)
    payload.update(request)


def validate_speculative_metrics(response: Any, feature: str) -> dict[str, Any]:
    label = feature.upper()
    timings = response.get("timings") if isinstance(response, dict) else None
    if not isinstance(timings, dict):
        fail(f"{label} smoke response did not contain timing metrics")
    drafted = timings.get("draft_n")
    accepted = timings.get("draft_n_accepted")
    if not isinstance(drafted, int) or drafted <= 0:
        fail(f"{label} smoke response reported no drafted tokens")
    if not isinstance(accepted, int) or accepted <= 0 or accepted > drafted:
        fail(f"{label} smoke response reported no valid accepted-token count")
    return {
        "draft_n": drafted,
        "draft_n_accepted": accepted,
        "predicted_per_second": timings.get("predicted_per_second"),
    }


def validate_mtp_metrics(response: Any) -> dict[str, Any]:
    return validate_speculative_metrics(response, "mtp")


def speech_smoke(config: Config, catalog: Catalog, profile: dict[str, Any], port: int) -> dict[str, Any]:
    model = catalog.models[profile["model"]]
    health = http_json(f"http://127.0.0.1:{port}/healthz", timeout=10)
    if not isinstance(health, dict) or health.get("status") != "ready":
        fail("speech health response is not ready")
    expected_revision = model["download"]["revision"]
    if health.get("model_revision") != expected_revision:
        fail("speech service is not running the catalog-pinned model revision")
    if not health.get("hip") or not health.get("device"):
        fail("speech service did not confirm its ROCm/HIP device")

    audio_path = ensure_speech_test_audio(config)
    audio = audio_path.read_bytes()
    boundary = "----halo-ai-seamless-smoke"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="target_lang"\r\n\r\nspa\r\n',
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="audio"; filename="input1.wav"\r\n',
        b"Content-Type: audio/wav\r\n\r\n",
        audio,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/v1/translate",
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "audio/wav",
        },
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            output = response.read()
            status = response.status
            inference_seconds = response.headers.get("X-Halo-AI-Inference-Seconds")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as exc:
        fail(f"speech smoke translation failed: {exc}")
    elapsed = time.monotonic() - started
    if status >= 300 or len(output) <= 44 or output[:4] != b"RIFF" or output[8:12] != b"WAVE":
        fail("speech smoke response was not a valid non-empty WAV")
    return {
        "health": health,
        "input": str(audio_path),
        "target_lang": "spa",
        "output_bytes": len(output),
        "inference_seconds": float(inference_seconds) if inference_seconds else None,
        "request_seconds": round(elapsed, 3),
        "wav_valid": True,
    }


def command_test(config: Config, catalog: Catalog, args: argparse.Namespace) -> int:
    smoke_hardware = hardware_snapshot(config)
    profile_id = args.profile_id
    if not profile_id:
        active_path = state_path(config, "active-profile.json")
        if not active_path.exists():
            fail("no active profile; specify one")
        profile_id = json.loads(read_text(active_path))["profile"]
    profile = catalog.profiles.get(profile_id)
    if not profile:
        fail(f"unknown profile: {profile_id}")
    preset = None
    if args.preset:
        preset = load_presets(config).get(args.preset)
        if not preset:
            fail(f"unknown preset: {args.preset}")
    port = port_for(config, profile["engine"])
    if profile["engine"] == "speech":
        if preset is not None:
            fail("request presets do not apply to speech profiles")
        result = speech_smoke(config, catalog, profile, port)
        last = state_path(config, "last-trial.json")
        if last.exists():
            trial = json.loads(read_text(last))
            if trial.get("profile") == profile_id and trial.get("status") in {"loaded", "success"}:
                smoke_tests = trial.setdefault("smoke_tests", [])
                if not isinstance(smoke_tests, list):
                    fail("last trial smoke_tests record is corrupt")
                smoke_tests.append({
                    "tested_at": utc_now(),
                    "kind": "speech-to-speech",
                    "target_lang": result["target_lang"],
                    "inference_seconds": result["inference_seconds"],
                    "output_bytes": result["output_bytes"],
                    "hardware": smoke_hardware,
                })
                trial.update({"status": "success", "smoke_tested_at": utc_now()})
                atomic_json(last, trial)
        print(json.dumps(result, indent=2))
        return 0
    if profile["engine"] == "lemonade":
        model_name = exact_lemonade_model(config, catalog, profile, port)
    else:
        models = http_json(f"http://127.0.0.1:{port}/v1/models")
        candidates = models.get("data", models) if isinstance(models, dict) else models
        if not candidates:
            fail("service returned no models")
        model_name = (candidates[0].get("id") or candidates[0].get("name")) if isinstance(candidates[0], dict) else str(candidates[0])
    is_vision = "vision" in profile.get("features", [])
    expected = "red" if is_vision else "halo-ai smoke test passed"
    model = catalog.models[profile["model"]]
    # DeepSeek V4's native template emits a separate reasoning_content field
    # even when the answer itself follows the requested exact form. Unlike the
    # pinned Qwen non-thinking template, that is expected behavior here.
    thinking = (
        profile.get("thinking")
        if "thinking" in profile
        else profile["engine"] == "llamacpp" and model.get("architecture") == "deepseek4"
    )
    prompt: Any = f"Reply with exactly: {expected}"
    if is_vision:
        # A tiny valid red PNG exercises Lemonade's image-content bridge and
        # llama.cpp's mmproj without introducing a mutable test asset.
        import base64
        import struct
        import zlib
        width = height = 16
        raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
        def chunk(kind: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
        prompt = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
            {"type": "text", "text": "What is the predominant color of this image? Reply with exactly: red"},
        ]
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 256,
    }
    if profile.get("thinking") is False:
        payload["reasoning_effort"] = "none"
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    elif profile.get("chat_template") == "nonthinking" or profile["engine"] == "ds4":
        payload["reasoning_effort"] = "none"
    if profile["engine"] == "rocmfpx":
        payload.update({"temperature": 0, "seed": 1})
    if preset:
        merge_request_preset(payload, preset["request"])
        thinking = (
            payload.get("reasoning_effort") != "none"
            and bool(payload.get("chat_template_kwargs", {}).get("enable_thinking", False))
        )
    if profile["engine"] == "lemonade":
        validate_lemonade_reasoning_template(container_name("lemonade"), payload, thinking)
    response = http_json(f"http://127.0.0.1:{port}/v1/chat/completions", method="POST", payload=payload, timeout=180)
    validate_smoke_response(response, expected, thinking)
    speculative_feature = next(
        (feature for feature in ("mtp", "dspark") if feature in profile.get("features", [])),
        None,
    )
    speculative_metrics = (
        validate_speculative_metrics(response, speculative_feature)
        if speculative_feature is not None else None
    )
    last = state_path(config, "last-trial.json")
    if last.exists():
        trial = json.loads(read_text(last))
        if trial.get("profile") == profile_id and trial.get("status") in {"loaded", "success"}:
            smoke_record = {
                "tested_at": utc_now(),
                "preset": args.preset or "default",
                "thinking": thinking,
                "backend_fingerprint": response.get("system_fingerprint") if isinstance(response, dict) else None,
                "hardware": smoke_hardware,
            }
            if speculative_metrics is not None:
                smoke_record["speculative"] = {
                    "feature": speculative_feature,
                    **speculative_metrics,
                }
            smoke_tests = trial.setdefault("smoke_tests", [])
            if not isinstance(smoke_tests, list):
                fail("last trial smoke_tests record is corrupt")
            smoke_tests.append(smoke_record)
            trial.update({
                "status": "success",
                "smoke_tested_at": utc_now(),
                "backend_fingerprint": response.get("system_fingerprint") if isinstance(response, dict) else None,
            })
            atomic_json(last, trial)
    print(json.dumps(response, indent=2))
    return 0


def command_env(config: Config) -> int:
    for key, value in hardware_snapshot(config).items():
        print(f"{key}={shlex.quote(str(value))}")
    return 0


def longbench_dataset_path(config: Config) -> Path:
    return (
        config.path("HALO_AI_CACHE_DIR") / "benchmarks" / "longbench-v2"
        / longbench.DATASET_REVISION / longbench.DATASET_FILENAME
    )


def command_bench_download(config: Config, args: argparse.Namespace) -> int:
    destination = Path(args.dataset).expanduser().resolve() if args.dataset else longbench_dataset_path(config)
    last_report = -1

    def progress(copied: int, total: int) -> None:
        nonlocal last_report
        bucket = copied // (64 * 1024**2)
        if bucket != last_report or copied == total:
            last_report = bucket
            print(f"LongBench-v2 download: {copied / 1024**2:.0f}/{total / 1024**2:.0f} MiB", flush=True)

    try:
        path = longbench.download_dataset(destination, progress)
    except longbench.LongBenchError as exc:
        fail(str(exc))
    print(json.dumps({
        "dataset": str(path),
        "repository": longbench.DATASET_REPOSITORY,
        "revision": longbench.DATASET_REVISION,
        "bytes": longbench.DATASET_BYTES,
        "sha256": longbench.DATASET_SHA256,
    }, indent=2))
    return 0


def longbench_request(url: str, payload: dict[str, Any], timeout: int = 600) -> Any:
    """Retry transient benchmark requests without hiding a persistent failure."""
    error = ""
    for attempt in range(1, 4):
        try:
            return http_json(url, method="POST", payload=payload, timeout=timeout)
        except HaloError as exc:
            error = str(exc)
            if attempt < 3:
                time.sleep(attempt)
    fail(f"LongBench-v2 request failed after 3 attempts: {error}")


def ds4_recent_timings(container: str, usage: Any) -> dict[str, Any] | None:
    """Translate ds4's latest completed request log into llama.cpp timing keys."""
    if not isinstance(usage, dict):
        return None
    prompt_n = usage.get("prompt_tokens")
    predicted_n = usage.get("completion_tokens")
    if not isinstance(prompt_n, int) or prompt_n < 1 or not isinstance(predicted_n, int) or predicted_n < 1:
        return None
    details = usage.get("prompt_tokens_details")
    cached_n = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    if not isinstance(cached_n, int) or not 0 <= cached_n < prompt_n:
        cached_n = 0
    processed_n = prompt_n - cached_n
    result = podman(["logs", container], capture=True, check=False)
    # Podman implementations may route container stderr through either captured
    # stream. ds4 writes its timing log to stderr, so normalize both.
    stdout = result.stdout if isinstance(result.stdout, str) else ""
    stderr = result.stderr if isinstance(result.stderr, str) else ""
    logs = stdout + "\n" + stderr
    prompt_matches = re.findall(r"\bprompt done ([0-9]+(?:\.[0-9]+)?)s", logs)
    generation_matches = re.findall(
        r"\bgen=([0-9]+)\b[^\n]*?\bavg=([0-9]+(?:\.[0-9]+)?) t/s ([0-9]+(?:\.[0-9]+)?)s",
        logs,
    )
    generation = next(
        (match for match in reversed(generation_matches) if int(match[0]) == predicted_n),
        None,
    )
    if not prompt_matches or generation is None:
        return None
    prompt_seconds = float(prompt_matches[-1])
    predicted_seconds = float(generation[2])
    if prompt_seconds <= 0 or predicted_seconds <= 0:
        return None
    return {
        "prompt_n": processed_n,
        "prompt_total_n": prompt_n,
        "prompt_cached_n": cached_n,
        "prompt_ms": round(prompt_seconds * 1000, 3),
        "prompt_per_token_ms": prompt_seconds * 1000 / processed_n,
        "prompt_per_second": processed_n / prompt_seconds,
        "predicted_n": predicted_n,
        "predicted_ms": round(predicted_seconds * 1000, 3),
        "predicted_per_token_ms": predicted_seconds * 1000 / predicted_n,
        "predicted_per_second": float(generation[1]),
        "source": "ds4_server_log",
    }


def lemonade_backend_port(container: str) -> int:
    processes = podman(["top", container, "args"], capture=True).stdout
    ports = set(re.findall(r"/llama-server\b[^\n]*?--port\s+([0-9]+)\b", processes))
    if len(ports) != 1:
        fail("could not identify exactly one active Lemonade llama.cpp backend port")
    return int(next(iter(ports)))


def longbench_input_tokens(
    base_url: str, model_name: str, prompt: str, *, lemonade_container: str | None = None,
) -> int:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning_effort": "none",
    }
    if lemonade_container is None:
        response = longbench_request(
            f"{base_url}/v1/chat/completions/input_tokens", payload, timeout=180,
        )
    else:
        port = lemonade_backend_port(lemonade_container)
        error = ""
        response = None
        for attempt in range(1, 4):
            try:
                result = podman([
                    "exec", "-i", lemonade_container, "curl", "-fsS", "-X", "POST",
                    f"http://127.0.0.1:{port}/v1/chat/completions/input_tokens",
                    "-H", "Content-Type: application/json", "--data-binary", "@-",
                ], capture=True, input_text=json.dumps(payload, separators=(",", ":")))
                response = json.loads(result.stdout)
                break
            except (HaloError, json.JSONDecodeError) as exc:
                error = str(exc)
                if attempt < 3:
                    time.sleep(attempt)
        if response is None:
            fail(f"active Lemonade llama.cpp token counter failed after 3 attempts: {error}")
    count = response.get("input_tokens") if isinstance(response, dict) else None
    if not isinstance(count, int) or count < 1:
        fail("active backend does not provide a valid /v1/chat/completions/input_tokens response")
    return count


def longbench_output_path(config: Config, profile_id: str, args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output).expanduser().resolve()
    suite_name = f"sample-{args.sample_id}" if args.sample_id else args.suite
    name = f"{profile_id}-{suite_name}-{args.overflow}-{longbench.DATASET_REVISION[:12]}.jsonl"
    return config.path("HALO_AI_STATE_DIR") / "benchmarks" / "longbench-v2" / name


def command_bench_run(config: Config, catalog: Catalog, args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit < 1:
        fail("--limit must be at least 1")
    if not 1 <= args.max_tokens <= 4096:
        fail("--max-tokens must be in [1, 4096]")
    profile = catalog.profiles.get(args.profile_id)
    if not profile:
        fail(f"unknown profile: {args.profile_id}")
    if profile["engine"] == "speech":
        fail("LongBench-v2 applies to LLM profiles, not speech profiles")
    active_path = state_path(config, "active-profile.json")
    if not active_path.exists():
        fail(f"start {profile['id']} before running LongBench-v2")
    try:
        active = json.loads(read_text(active_path))
    except json.JSONDecodeError:
        fail(f"active profile record is corrupt: {active_path}")
    if active.get("profile") != profile["id"]:
        fail(f"active profile is {active.get('profile')}; start {profile['id']} first")
    port = port_for(config, profile["engine"])
    base_url = f"http://127.0.0.1:{port}"
    if profile["engine"] == "lemonade":
        model_name = exact_lemonade_model(config, catalog, profile, port)
        token_container = container_name("lemonade")
    else:
        models = http_json(f"{base_url}/v1/models")
        candidates = models.get("data", models) if isinstance(models, dict) else models
        if not candidates:
            fail("active benchmark service returned no models")
        model_name = candidates[0].get("id") if isinstance(candidates[0], dict) else str(candidates[0])
        token_container = None
    # Refuse an hours-long run before proving the exact chat-template-aware
    # token-count endpoint is reachable. ds4 currently exposes completions but
    # no tokenizer route; allow only one explicitly named sample there, then
    # replace the conservative estimate with response usage.
    exact_counter = profile["engine"] != "ds4"
    if exact_counter:
        longbench_input_tokens(
            base_url, model_name, "LongBench-v2 token counter canary",
            lemonade_container=token_container,
        )
    elif args.sample_id is None:
        fail("ds4 has no token-count endpoint; use --sample-id for one bounded sample")
    elif args.overflow != "fit":
        fail("ds4 cannot safely middle-truncate without a token-count endpoint")

    dataset_path = Path(args.dataset).expanduser().resolve() if args.dataset else longbench_dataset_path(config)
    if not dataset_path.exists():
        command_bench_download(config, argparse.Namespace(dataset=str(dataset_path)))
    try:
        items = longbench.load_dataset(dataset_path)
        selected = longbench.select_items(
            items, suite=args.suite, limit=args.limit, length=args.length,
            difficulty=args.difficulty, domain=args.domain, sample_id=args.sample_id,
        )
    except longbench.LongBenchError as exc:
        fail(str(exc))
    if not selected:
        fail("LongBench-v2 filters selected no samples")

    output = longbench_output_path(config, profile["id"], args)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    run_spec = {
        "schema_version": 1,
        "benchmark": "LongBench-v2",
        "dataset_repository": longbench.DATASET_REPOSITORY,
        "dataset_revision": longbench.DATASET_REVISION,
        "dataset_sha256": longbench.DATASET_SHA256,
        "profile": profile["id"],
        "profile_context_tokens": profile["context"],
        "model": model_name,
        "suite": args.suite,
        "overflow": args.overflow,
        "max_output_tokens": args.max_tokens,
        "length": args.length,
        "difficulty": args.difficulty,
        "domain": args.domain,
        "sample_id": args.sample_id,
        "limit": args.limit,
        "sample_ids": [str(item["_id"]) for item in selected],
    }
    if manifest_path.exists():
        try:
            prior_manifest = json.loads(read_text(manifest_path))
        except json.JSONDecodeError:
            fail(f"benchmark manifest is corrupt: {manifest_path}")
        if prior_manifest.get("run") != run_spec:
            fail(f"benchmark output belongs to a different run: {output}")
    else:
        trial_path = state_path(config, "last-trial.json")
        trial = json.loads(read_text(trial_path)) if trial_path.exists() else {}
        atomic_json(manifest_path, {
            "created_at": utc_now(), "run": run_spec,
            "backend": trial.get("backend"), "image": trial.get("image"),
            "image_digest": trial.get("image_digest"),
            "trial_fingerprint": trial.get("fingerprint"),
        }, mode=0o644)

    try:
        existing = longbench.load_jsonl(output) if output.exists() else []
    except longbench.LongBenchError as exc:
        fail(str(exc))
    completed_ids = {str(item.get("_id")) for item in existing}
    remaining = [item for item in selected if str(item["_id"]) not in completed_ids]
    budget = profile["context"] - args.max_tokens
    if budget < 1:
        fail("max output tokens leave no room for the benchmark prompt")
    print(
        f"LongBench-v2: {len(selected)} selected, {len(existing)} recorded, "
        f"{len(remaining)} remaining; output {output}",
        flush=True,
    )

    for position, item in enumerate(remaining, 1):
        prompt = longbench.render_prompt(item)
        if exact_counter:
            input_tokens = longbench_input_tokens(
                base_url, model_name, prompt, lemonade_container=token_container,
            )
            token_count_method = "backend_input_tokens"
        else:
            # Deliberately conservative for the explicitly selected sample.
            # The authoritative value is taken from response usage below.
            input_tokens = (len(prompt.encode("utf-8")) + 1) // 2 + 512
            token_count_method = "preflight_estimate_bytes_over_2_plus_512"
        truncated = False
        if input_tokens > budget:
            if args.overflow == "fit":
                record = {
                    "schema_version": 1, "_id": str(item["_id"]),
                    "status": "skipped_overflow", "input_tokens": input_tokens,
                    "token_budget": budget, "domain": item["domain"],
                    "sub_domain": item["sub_domain"], "difficulty": item["difficulty"],
                    "length": item["length"], "recorded_at": utc_now(),
                }
                longbench.append_jsonl(output, record)
                print(f"[{position}/{len(remaining)}] {item['_id']}: skipped ({input_tokens} > {budget})", flush=True)
                continue
            try:
                prompt, input_tokens = longbench.truncate_to_budget(
                    item, budget,
                    lambda value: longbench_input_tokens(
                        base_url, model_name, value, lemonade_container=token_container,
                    ),
                )
            except longbench.LongBenchError as exc:
                fail(f"sample {item['_id']}: {exc}")
            truncated = True

        before = hardware_snapshot(config)
        started = time.monotonic()
        response = longbench_request(
            f"{base_url}/v1/chat/completions",
            {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": args.max_tokens,
                "stream": False,
                "reasoning_effort": "none",
            },
        )
        elapsed = time.monotonic() - started
        try:
            message = response["choices"][0]["message"]
            answer_text = message.get("content") or ""
        except (KeyError, IndexError, TypeError):
            fail(f"sample {item['_id']} returned an invalid chat completion")
        prediction = longbench.extract_answer(answer_text)
        usage = response.get("usage") if isinstance(response, dict) else None
        reported_prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if not exact_counter:
            if not isinstance(reported_prompt_tokens, int) or reported_prompt_tokens < 1:
                fail(f"sample {item['_id']} returned no authoritative ds4 prompt token count")
            input_tokens = reported_prompt_tokens
            token_count_method = "response_usage"
        after = hardware_snapshot(config)
        timings = response.get("timings") if isinstance(response, dict) else None
        if timings is None and profile["engine"] == "ds4":
            timings = ds4_recent_timings(container_name("ds4"), usage)
        record = {
            "schema_version": 1, "_id": str(item["_id"]), "status": "complete",
            "domain": item["domain"], "sub_domain": item["sub_domain"],
            "difficulty": item["difficulty"], "length": item["length"],
            "answer": item["answer"], "response": answer_text,
            "pred": prediction, "judge": prediction == item["answer"],
            "input_tokens": input_tokens, "token_budget": budget,
            "token_count_method": token_count_method,
            "truncated": truncated, "elapsed_seconds": round(elapsed, 3),
            "usage": usage,
            "timings": timings,
            "backend_fingerprint": response.get("system_fingerprint") if isinstance(response, dict) else None,
            "gtt_used_bytes_before": before["HALO_AI_GTT_USED_BYTES"],
            "gtt_used_bytes_after": after["HALO_AI_GTT_USED_BYTES"],
            "cpu_mem_available_bytes_before": before["HALO_AI_CPU_MEM_AVAILABLE_BYTES"],
            "cpu_mem_available_bytes_after": after["HALO_AI_CPU_MEM_AVAILABLE_BYTES"],
            "recorded_at": utc_now(),
        }
        longbench.append_jsonl(output, record)
        print(
            f"[{position}/{len(remaining)}] {item['_id']}: pred={prediction or '-'} "
            f"answer={item['answer']} tokens={input_tokens} truncated={truncated} {elapsed:.1f}s",
            flush=True,
        )

    rows = longbench.load_jsonl(output)
    score = longbench.score_records(rows)
    atomic_json(output.with_suffix(output.suffix + ".score.json"), score, mode=0o644)
    print(json.dumps(score, indent=2))
    return 0


def command_bench_score(args: argparse.Namespace) -> int:
    path = Path(args.results).expanduser().resolve()
    try:
        score = longbench.score_records(longbench.load_jsonl(path))
    except longbench.LongBenchError as exc:
        fail(str(exc))
    print(json.dumps(score, indent=2))
    return 0


def command_bench(config: Config, catalog: Catalog, args: argparse.Namespace) -> int:
    if args.bench_action == "download":
        return command_bench_download(config, args)
    if args.bench_action == "run":
        return command_bench_run(config, catalog, args)
    return command_bench_score(args)


def inventory_full_verification(
    config: Config, model: dict[str, Any],
) -> dict[str, Any] | None:
    """Return reusable full-verification evidence when it still matches the catalog."""
    inventory_path = config.path("HALO_AI_INVENTORY_FILE")
    if not inventory_path.exists():
        return None
    try:
        document = json.loads(read_text(inventory_path))
    except json.JSONDecodeError:
        return None
    expected = {
        str(path): entry
        for entry, path in model_paths(config, model)
    }
    for result in document.get("results", []) if isinstance(document, dict) else []:
        if (
            not isinstance(result, dict)
            or result.get("model") != model["id"]
            or result.get("full") is not True
            or result.get("valid") is not True
        ):
            continue
        files = result.get("files")
        if not isinstance(files, list) or {item.get("path") for item in files} != set(expected):
            continue
        reusable = True
        for item in files:
            entry = expected[item["path"]]
            path = Path(item["path"])
            try:
                info = path.stat(follow_symlinks=False)
            except OSError:
                reusable = False
                break
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_size != entry["bytes"]
                or item.get("expected_bytes") != entry["bytes"]
                or item.get("sha256") != entry["sha256"]
                or item.get("digest_valid") is not True
            ):
                reusable = False
                break
        if reusable:
            return result
    return None


def write_optional_json(path_value: str | None, document: dict[str, Any]) -> None:
    if path_value:
        destination = Path(path_value).expanduser().resolve()
        atomic_json(destination, document, mode=0o644)


def command_tune(config: Config, catalog: Catalog, args: argparse.Namespace) -> int:
    if args.tune_action == "mtp-compare":
        try:
            result = rocmfpx_tune.compare_mtp_records(
                rocmfpx_tune.load_json(Path(args.baseline).expanduser().resolve()),
                rocmfpx_tune.load_json(Path(args.candidate).expanduser().resolve()),
            )
        except rocmfpx_tune.TuneError as exc:
            fail(str(exc))
        write_optional_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 0
    if args.tune_action == "stage3-preflight":
        return command_tune_stage3_preflight(config, catalog, args)
    if args.tune_action == "stage3-capture-target":
        return command_tune_stage3_capture_target(config, catalog, args)
    if args.tune_action == "stage3-capture-draft":
        return command_tune_stage3_capture_draft(config, catalog, args)
    if args.tune_action == "stage3-score":
        try:
            result = rocmfpx_tune.score_stage3_records(
                rocmfpx_tune.load_json(Path(args.results).expanduser().resolve()),
            )
        except rocmfpx_tune.TuneError as exc:
            fail(str(exc))
        write_optional_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 0
    pending = state_path(config, "pending-trial.env")
    active = state_path(config, "active-trial.json")
    if args.tune_action == "status":
        last = state_path(config, "last-trial.json")
        pending_data = parse_env_file(pending) if pending.exists() else None
        result = {
            "boot_id": boot_id(),
            "pending": pending_data,
            "pending_reboot_crossed": bool(pending_data) and pending_data.get("HALO_AI_PENDING_ORIGIN_BOOT_ID") != boot_id(),
            "active": json.loads(read_text(active)) if active.exists() else None,
            "last": json.loads(read_text(last)) if last.exists() else None,
        }
        print(json.dumps(result, indent=2))
    else:
        if pending.exists():
            archive = state_path(config, f"discarded-pending-{int(time.time())}.env")
            os.replace(pending, archive)
            print(f"Archived pending adjustment to {archive}")
        else:
            print("No pending adjustment.")
    return 0


def command_host_profile(args: argparse.Namespace) -> int:
    if args.host_action == "status":
        print(json.dumps(host_profile.persistent_status(), indent=2))
        return 0
    try:
        def invoke() -> int:
            if args.host_action == "init":
                return host_profile.initialize(dry_run=args.dry_run, assume_yes=args.yes, no_snapshot=args.no_snapshot)
            if args.host_action == "set":
                return host_profile.mutate(
                    args.profile, gtt_gib=args.gtt_gib,
                    dry_run=args.dry_run, assume_yes=args.yes, no_snapshot=args.no_snapshot,
                )
            return host_profile.rollback(args.backup_id, dry_run=args.dry_run, assume_yes=args.yes, no_snapshot=args.no_snapshot)
        if args.dry_run:
            return invoke()
        with host_profile.mutation_lock():
            return invoke()
    except host_profile.HostProfileError as exc:
        fail(str(exc))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="halo-ai",
        description="Guarded lifecycle manager for AMD Strix Halo inference services.",
    )
    parser.add_argument("--config", help="use only this configuration file")
    parser.add_argument("--version", action="version", version=VERSION)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    models = sub.add_parser("models").add_subparsers(dest="models_action", required=True)
    models.add_parser("scan")
    models.add_parser("list")
    show_model = models.add_parser("show"); show_model.add_argument("model_id")
    download_model = models.add_parser("download"); download_model.add_argument("model_id"); download_model.add_argument("--dry-run", action="store_true")
    verify = models.add_parser("verify"); verify.add_argument("model_id", nargs="?"); verify.add_argument("--full", action="store_true")
    profiles = sub.add_parser("profiles").add_subparsers(dest="profiles_action", required=True)
    profiles.add_parser("list")
    acquire_profile = profiles.add_parser("acquire"); acquire_profile.add_argument("profile_id"); acquire_profile.add_argument("--dry-run", action="store_true")
    show_profile = profiles.add_parser("show"); show_profile.add_argument("profile_id")
    render_profile = profiles.add_parser("render"); render_profile.add_argument("profile_id")
    presets = sub.add_parser("presets").add_subparsers(dest="presets_action", required=True)
    presets.add_parser("list")
    show_preset = presets.add_parser("show"); show_preset.add_argument("preset_id")
    render_preset = presets.add_parser("render"); render_preset.add_argument("preset_id")
    install = sub.add_parser("install"); install.add_argument("engine", choices=["lemonade", "llamacpp", "rocmfpx", "ds4", "speech", "vllm", "all"], nargs="?", default="all")
    start = sub.add_parser("start", help="start one catalog profile"); start.add_argument("profile_id"); start.add_argument("--switch", action="store_true")
    stop = sub.add_parser(
        "stop",
        help="stop all managed runtimes (or the runtime for one profile)",
        description=(
            "Stop halo-ai runtime containers, including an in-progress model load. "
            "The default target is all. Images, models, caches, and volumes are preserved."
        ),
    )
    stop.add_argument("target", nargs="?", default="all", help="profile ID or 'all' (default: all)")
    restart = sub.add_parser("restart", help="stop and start one catalog profile"); restart.add_argument("profile_id")
    sub.add_parser("status")
    sub.add_parser("env")
    logs = sub.add_parser("logs"); logs.add_argument("profile_id", nargs="?"); logs.add_argument("-f", "--follow", action="store_true")
    test = sub.add_parser("test"); test.add_argument("profile_id", nargs="?"); test.add_argument("--preset")
    bench = sub.add_parser("bench", help="run reproducible inference benchmarks").add_subparsers(dest="bench_name", required=True)
    longbench_v2 = bench.add_parser("longbench-v2", help="pinned LongBench-v2 multiple-choice benchmark")
    longbench_actions = longbench_v2.add_subparsers(dest="bench_action", required=True)
    bench_download = longbench_actions.add_parser("download", help="download and verify the pinned dataset")
    bench_download.add_argument("--dataset", help="alternate destination for the pinned data.json")
    bench_run = longbench_actions.add_parser("run", help="run or resume against an active profile")
    bench_run.add_argument("profile_id")
    bench_run.add_argument("--suite", choices=["canary", "full"], default="canary")
    bench_run.add_argument(
        "--overflow", choices=["fit", "middle"], default="fit",
        help="skip over-context samples (fit) or explicitly middle-truncate documents (middle)",
    )
    bench_run.add_argument("--max-tokens", type=int, default=128)
    bench_run.add_argument("--limit", type=int)
    bench_run.add_argument("--length", choices=["short", "medium", "long"])
    bench_run.add_argument("--difficulty", choices=["easy", "hard"])
    bench_run.add_argument("--domain")
    bench_run.add_argument("--sample-id", help="run exactly one pinned dataset sample ID")
    bench_run.add_argument("--dataset", help="alternate path to the exact pinned data.json")
    bench_run.add_argument("--output", help="resumable JSONL output path")
    bench_score = longbench_actions.add_parser("score", help="score an existing halo-ai JSONL run")
    bench_score.add_argument("results")
    update = sub.add_parser("update"); update.add_argument("engine", choices=["lemonade", "llamacpp", "rocmfpx", "ds4", "speech", "vllm", "all"])
    tune = sub.add_parser("tune", help="inspect guarded trials and score optimization evidence").add_subparsers(dest="tune_action", required=True)
    tune.add_parser("status")
    tune.add_parser("discard")
    mtp_compare = tune.add_parser(
        "mtp-compare", help="compare same-host ROCmFPX baseline and MTP records",
    )
    mtp_compare.add_argument("baseline")
    mtp_compare.add_argument("candidate")
    mtp_compare.add_argument("--output", help="optional atomic JSON report path")
    stage3_preflight = tune.add_parser(
        "stage3-preflight", help="no-download inventory gate for the Qwen3.6 proxy screen",
    )
    stage3_preflight.add_argument(
        "--full", action="store_true",
        help="SHA-256 verify cached target/proxy artifacts that lack reusable full evidence",
    )
    stage3_target = tune.add_parser(
        "stage3-capture-target",
        help="capture greedy target tokens from Qwen3.8-owned rendered prefixes",
    )
    stage3_target.add_argument("output")
    stage3_target.add_argument("--suite", choices=["canary", "full"], default="canary")
    stage3_target.add_argument(
        "--mode", choices=["all", "nonthinking", "thinking"], default="all",
    )
    stage3_target.add_argument(
        "--context-bucket", choices=["all", "short", "4k", "32k"], default="all",
    )
    stage3_target.add_argument("--target-tokens", type=int, default=12)
    stage3_draft = tune.add_parser(
        "stage3-capture-draft",
        help="feed exact captured target prefixes to the cached Qwen3.6 proxy",
    )
    stage3_draft.add_argument("target_capture")
    stage3_draft.add_argument("output")
    stage3_score = tune.add_parser(
        "stage3-score", help="score exact-prefix token proposal captures",
    )
    stage3_score.add_argument("results")
    stage3_score.add_argument("--output", help="optional atomic JSON report path")
    host = sub.add_parser("host-profile").add_subparsers(dest="host_action", required=True)
    host.add_parser("status")
    host_init = host.add_parser("init")
    host_set = host.add_parser("set"); host_set.add_argument("profile", choices=["gpu", "npu"])
    host_set.add_argument("--gtt-gib", type=int, choices=sorted(host_profile.GTT_SETTINGS))
    host_rollback = host.add_parser("rollback"); host_rollback.add_argument("backup_id")
    for host_mutation in (host_init, host_set, host_rollback):
        host_mutation.add_argument("--dry-run", action="store_true")
        host_mutation.add_argument("--yes", action="store_true")
        host_mutation.add_argument("--no-snapshot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    catalog = load_catalog(config)
    if args.command in {"install", "update", "start", "stop", "restart", "status", "logs", "test", "bench"} or (
        args.command == "models" and args.models_action == "download"
    ) or (
        args.command == "profiles" and args.profiles_action == "acquire"
    ) or (
        args.command == "tune"
        and args.tune_action in {"stage3-capture-target", "stage3-capture-draft"}
    ):
        assert_operator(config)
    if args.command == "doctor": return command_doctor(config, catalog, args)
    if args.command == "models": return command_models(config, catalog, args)
    if args.command == "profiles": return command_profiles(config, catalog, args)
    if args.command == "presets": return command_presets(config, args)
    if args.command in {"install", "update"}:
        if args.engine == "all":
            engines = [engine for engine, key in ENGINE_IMAGE_KEYS.items() if config.get(key)]
            skipped = [engine for engine, key in ENGINE_IMAGE_KEYS.items() if not config.get(key)]
            if skipped:
                print(f"Skipping deferred engines without configured images: {', '.join(skipped)}")
        else:
            engines = [args.engine]
        return command_runtime_install(config, engines) if args.command == "install" else command_update(config, engines)
    if args.command == "start": return command_start(config, catalog, args)
    if args.command == "stop": return command_stop(config, catalog, args)
    if args.command == "restart":
        command_stop(config, catalog, argparse.Namespace(target=args.profile_id))
        return command_start(config, catalog, argparse.Namespace(profile_id=args.profile_id, switch=False))
    if args.command == "status": return command_status(config, catalog, args)
    if args.command == "env": return command_env(config)
    if args.command == "logs": return command_logs(catalog, args)
    if args.command == "test": return command_test(config, catalog, args)
    if args.command == "bench": return command_bench(config, catalog, args)
    if args.command == "tune": return command_tune(config, catalog, args)
    if args.command == "host-profile": return command_host_profile(args)
    fail(f"unhandled command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HaloError as exc:
        eprint(f"error: {exc}")
        raise SystemExit(2)
    except KeyboardInterrupt:
        eprint("error: interrupted")
        raise SystemExit(130)
