"""Explicit engine metadata and the small policy hooks shared by catalog consumers."""
from dataclasses import dataclass
from typing import Any, Callable

import engine_halogen


@dataclass(frozen=True)
class EngineSpec:
    image_key: str
    port_key: str
    container: str
    health_path: str = "/v1/models"
    validate_profile: Callable[[dict[str, Any], dict[str, Any]], None] | None = None
    artifact_roles: Callable[[dict[str, Any]], set[str]] | None = None


REGISTRY = {
    "lemonade": EngineSpec("LEMONADE_IMAGE", "LEMONADE_PORT", "halo-lemonade", "/live"),
    "llamacpp": EngineSpec("LLAMACPP_IMAGE", "LLAMACPP_PORT", "halo-llamacpp"),
    "rocmfpx": EngineSpec("ROCMFPX_IMAGE", "ROCMFPX_PORT", "halo-rocmfpx"),
    "strixvulkan": EngineSpec("STRIXVULKAN_IMAGE", "STRIXVULKAN_PORT", "halo-strixvulkan"),
    "strixvulkan075": EngineSpec("STRIXVULKAN075_IMAGE", "STRIXVULKAN_PORT", "halo-strixvulkan075"),
    "halogen": EngineSpec("HALOGEN_IMAGE", "HALOGEN_API_PORT", "halo-halogen", "/health",
                          engine_halogen.validate_profile, engine_halogen.required_roles),
    "ds4": EngineSpec("DS4_IMAGE", "DS4_PORT", "halo-ds4"),
    "speech": EngineSpec("SPEECH_IMAGE", "SPEECH_PORT", "halo-speech", "/healthz"),
    "vllm": EngineSpec("VLLM_IMAGE", "VLLM_PORT", "halo-vllm"),
}


def required_roles(profile: dict[str, Any]) -> set[str]:
    policy = REGISTRY[profile["engine"]].artifact_roles
    if policy is not None:
        return policy(profile)
    roles = {"main"}
    features = set(profile.get("features", []))
    if "mtp" in features:
        roles.add(profile.get("settings", {}).get("mtp_role", "mtp"))
    if "vision" in features:
        roles.add("mmproj")
    if "dspark" in features:
        roles.add("dspark")
    if profile.get("chat_template"):
        roles.add("chat_template")
    return roles
