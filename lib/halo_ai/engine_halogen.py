"""Halogen policy and pure rendering; host and network I/O stay with callers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from artifact_store import Artifact
from errors import fail
import halogen_npu


RELEASE = "0.16.2"
IMAGE_DIGEST = "sha256:0c61bf84ac22308a53f5d1ca6b86806702d7039e5ebc51cae4c66621b92fe04a"
IMAGE = f"ghcr.io/peonist-ai/halogen-flash-server@{IMAGE_DIGEST}"


def required_roles(profile: dict[str, Any]) -> set[str]:
    features = set(profile.get("features", []))
    return {"main", "processor", "ngram" if "external-ngram" in features else "overlay"} | ({"mmproj"} if "vision" in features else set())


def validate_profile(profile: dict[str, Any], model: dict[str, Any]) -> None:
    identifier = profile["id"]
    context = profile["context"]
    features = set(profile.get("features", []))
    settings = profile.get("settings", {})
    npu_models = settings.get("npu_models", [])
    if (
        not isinstance(npu_models, list)
        or any(not isinstance(name, str) or name not in halogen_npu.MODEL_IDS for name in npu_models)
        or len(npu_models) != len(set(npu_models))
        or ("npu" in features) != bool(npu_models)
    ):
        fail(f"profile {identifier} has invalid Halogen NPU models")
    integers = {
        "kv_pool_positions": (context, 262144), "parallel": (1, 2),
        "prefill_chunk": (1024, 16384), "prompt_cache": (0, 2),
        "drafter": (0, 1), "prompt_lookup": (0, 1), "max_tokens_default": (1, min(context, 65536)),
    }
    for key, (minimum, maximum) in integers.items():
        value = settings.get(key)
        if type(value) is not int or not minimum <= value <= maximum:
            fail(f"profile {identifier} has invalid Halogen setting {key}")
    if (
        model.get("format") != "hgn"
        or context > 262144
        or settings["prefill_chunk"] > context
        or settings.get("reasoning_effort") not in {"low", "medium", "xhigh"}
        or set(settings) != set(integers) | {"reasoning_effort"} | ({"npu_models"} if "npu" in features else set())
        or not features <= {"mtp", "vision", "external-ngram", "npu"}
        or ("mtp" in features) != bool(settings["drafter"])
        or (settings["prompt_lookup"] and not settings["drafter"])
        or len([f for f in model["files"] if f["role"] == "overlay"]) != (0 if "external-ngram" in features else 1)
        or len([f for f in model["files"] if f["role"] == "ngram"]) != (1 if "external-ngram" in features else 0)
        or len([f for f in model["files"] if f["role"] == "main"]) != 1
        or not any(f["role"] == "processor" for f in model["files"])
    ):
        fail(f"profile {identifier} violates the Halogen candidate policy")


def render_arguments(
    profile: dict[str, Any], model: dict[str, Any], selected: list[tuple[dict[str, Any], Path]],
    *, port: int, image: str, auxiliary: tuple[Artifact, ...], npu_root: Path | None,
    mounts: list[tuple[Path, str]],
) -> list[str]:
    command: list[str] = []
    command.extend(["--group-add", "keep-groups", "--ipc=host", "--ulimit", "memlock=-1:-1"])
    settings = profile["settings"]
    for entry, path in selected:
        relative = Path(entry["path"]).relative_to(model["repository"])
        command.extend(["--mount", f"type=bind,src={path},dst=/models/{relative},ro"])
    artifact_paths = {
        entry["role"]: f"/models/{Path(entry['path']).relative_to(model['repository'])}"
        for entry, _path in selected
    }
    environment = {
        "HALOGEN_API_PORT": port, "HALOGEN_BIND": "127.0.0.1",
        "HALOGEN_CHECKPOINT": artifact_paths["main"],
        "HALOGEN_TOKENIZER": "/models/tokenizer",
        "HALOGEN_MODEL_ID": model["id"], "HALOGEN_CTX": profile["context"],
        "HALOGEN_KV_POOL_POSITIONS": settings["kv_pool_positions"],
        "HALOGEN_KV_SLOTS": settings["parallel"],
        "HALOGEN_MAX_TOK": settings["prefill_chunk"],
        "HALOGEN_PROMPT_CACHE": settings["prompt_cache"],
        "HALOGEN_DRAFTER_DEFAULT": settings["drafter"],
        "HALOGEN_PLD": "3,3" if settings["prompt_lookup"] else "0",
        "HALOGEN_REASONING_EFFORT": settings["reasoning_effort"],
        "HALOGEN_MAX_TOKENS_DEFAULT": settings["max_tokens_default"],
        "HALOGEN_TEMPERATURE": "1.0", "HALOGEN_TOP_P": "0.95", "HALOGEN_TOP_K": "20",
    }
    if "external-ngram" in profile.get("features", []):
        environment["HALOGEN_NGRAM_TABLE"] = artifact_paths["ngram"]
    else:
        environment["HALOGEN_CK_OVERLAY"] = artifact_paths["overlay"]
    if "vision" in profile.get("features", []):
        environment["HALOGEN_VISION_TOWER"] = artifact_paths["mmproj"]
    if "npu" in profile.get("features", []):
        for artifact in auxiliary:
            relative = artifact.destination.relative_to(npu_root)
            command.extend(["--mount", f"type=bind,src={artifact.destination},dst=/models/npu/{relative},ro"])
        command.extend(["--device", "/dev/accel/accel0"])
        for source, destination in mounts:
            command.extend(["--mount", f"type=bind,src={source},dst={destination},ro"])
        environment["HALOGEN_NPU_MODELS"] = ",".join(settings["npu_models"])
    for key, value in sorted(environment.items()):
        command.extend(["-e", f"{key}={value}"])
    command.append(image)
    return command


def validate_health(profile: dict[str, Any], health: Any) -> None:
    settings = profile["settings"]
    expected = {
        "context": profile["context"], "slots": settings["parallel"],
        "kv_pool_positions": settings["kv_pool_positions"],
        "drafter_default": "mtp" if settings["drafter"] else "serial",
        "checkpoint_format": "hgn",
        "reasoning_effort_default": settings["reasoning_effort"],
    }
    if (
        not isinstance(health, dict)
        or any(health.get(key) != value for key, value in expected.items())
        or health.get("version") != {"api": RELEASE, "engine": RELEASE, "match": True}
        or not health.get("engine", {}).get("responds")
        or health.get("capability_probe") != "ok"
        or not health.get("drafter_weights_loaded")
        or health.get("chat_template", {}).get("probe") != "passed"
        or health.get("prompt_cache", {}).get("mode") != settings["prompt_cache"]
        or health.get("vision", {}).get("enabled") != ("vision" in profile.get("features", []))
    ):
        fail("Halogen health does not match the selected runtime/profile policy")
    if "npu" in profile.get("features", []):
        npu = health.get("npu", {})
        if npu.get("status") != "ok" or not set(settings["npu_models"]) <= set(npu.get("models", [])):
            fail("Halogen NPU engine is not healthy for the selected models")


def validate_advertised_models(profile: dict[str, Any], advertised: dict[str, Any]) -> None:
    identifiers = {item.get("id") for item in advertised.get("data", [])}
    if not set(profile["settings"]["npu_models"]) <= identifiers:
        fail("Halogen did not advertise every selected NPU model")


def resident_bytes(profile: dict[str, Any], auxiliary: tuple[Artifact, ...]) -> int:
    # HGN tables are demand-paged: disk size is not resident weight size.
    total = (68 + 14 + (2 if "vision" in profile.get("features", []) else 0)) * 1024**3
    total += profile["settings"]["kv_pool_positions"] * 32 * 1024
    return total + npu_memory_bytes(auxiliary)


def npu_memory_bytes(auxiliary: tuple[Artifact, ...]) -> int:
    return sum(a.bytes for a in auxiliary) + (2 * 1024**3 if auxiliary else 0)
