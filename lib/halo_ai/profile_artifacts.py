"""Resolve a profile's complete file selection independently of CLI and host readiness."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from artifact_store import Artifact
from errors import fail
import engines
import halogen_npu


def model_paths(root: Path, model: dict[str, Any], roles: set[str] | frozenset[str] | None = None) -> list[tuple[dict[str, Any], Path]]:
    root = root.resolve()
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


@dataclass(frozen=True)
class ModelSelection:
    model: dict[str, Any]
    roles: frozenset[str] | None
    files: tuple[tuple[dict[str, Any], Path], ...]

    def downloads(self) -> tuple[Artifact, ...]:
        download = self.model.get("download")
        if not isinstance(download, dict) or download.get("provider") != "huggingface":
            fail(f"model {self.model['id']} has no approved pinned downloader")
        allowed = set(download["allow_patterns"])
        artifacts = tuple(Artifact.from_entry(path, entry, download["repository"], download["revision"])
                          for entry, path in self.files)
        if any(artifact.source_path not in allowed for artifact in artifacts):
            fail(f"model {self.model['id']} selected a file outside its download allowlist")
        return artifacts


def select_model(root: Path, model: dict[str, Any], roles: set[str] | None) -> ModelSelection:
    # Transformers processors and weights form one directory-backed model.
    selected_roles = None if model.get("format") == "transformers" or roles is None else frozenset(roles)
    return ModelSelection(model, selected_roles, tuple(model_paths(root, model, selected_roles)))


@dataclass(frozen=True)
class ProfileArtifacts:
    primary: ModelSelection
    draft: ModelSelection | None
    auxiliary: tuple[Artifact, ...]
    auxiliary_root: Path | None

    @property
    def models(self) -> tuple[ModelSelection, ...]:
        return (self.primary, self.draft) if self.draft else (self.primary,)

    @property
    def weight_bytes(self) -> int:
        roles = {"main", "weights", "mtp", "mtp_q8"}
        return sum(entry["bytes"] for selection in self.models
                   for entry, _ in selection.files if entry["role"] in roles)


def resolve(root: Path, models: dict[str, dict[str, Any]], profile: dict[str, Any]) -> ProfileArtifacts:
    primary = select_model(root, models[profile["model"]], engines.required_roles(profile))
    features = set(profile.get("features", []))
    draft = select_model(root, models[profile["draft_model"]], {"main"}) if "dflash" in features else None
    auxiliary_root = None
    auxiliary: tuple[Artifact, ...] = ()
    if profile["engine"] == "halogen" and "npu" in features:
        auxiliary_root = root.resolve() / "halogen-npu" / halogen_npu.MANIFEST["release"]
        try:
            auxiliary_root.resolve().relative_to(root.resolve())
        except ValueError:
            fail(f"NPU artifact path escapes model root: {auxiliary_root}")
        auxiliary = halogen_npu.pinned_artifacts(auxiliary_root, profile["settings"]["npu_models"])
    return ProfileArtifacts(primary, draft, auxiliary, auxiliary_root)


def artifact_classes(profile: dict[str, Any], selection: ProfileArtifacts) -> tuple[list[str], list[str]]:
    """Describe the resolved selection, rather than a parallel list of expected files."""
    engine = profile["engine"]
    model = selection.primary.model
    roles = {entry["role"] for entry, _ in selection.primary.files}
    features = set(profile.get("features", []))
    vision = ["vision"] if "mmproj" in roles else []
    if engine == "halogen":
        selected = ["hgn"]
        if "overlay" in roles:
            selected.append("quality-overlay")
        if "ngram" in roles:
            selected.append("external-ngram")
        selected += ["tokenizer", "halogen-runtime"]
        if selection.auxiliary:
            selected.append("npu")
        if "mtp" in features:
            selected.append("embedded-mtp")
        selected += vision
    elif engine == "rocmfpx":
        quantization = "fp8" if model.get("quantization") == "Q8_0_ROCMFPX" else "fp4"
        selected = [quantization, "rocmfpx-runtime"]
        if "mtp" in features:
            selected.append("in-gguf-mtp")
    elif engine in {"strixvulkan", "strixvulkan075"} and model["id"] == "qwen3.8-27b-ud-q6-k-xl":
        selected = ["q6-xl"] + vision
        if selection.draft:
            selected.append("dflash2")
        selected.append("strixvulkan-runtime")
    else:
        selected = ["model"]
        if roles & {"mtp", "mtp_q8"}:
            selected.append("mtp-sidecar")
        selected += vision
        if selection.draft:
            selected.append("dflash2")
        selected.append("runtime")
    if engine == "rocmfpx":
        excluded = ["fp4" if "fp8" in selected else "fp8", "npu", "vision", "bf16", "reference"]
    elif engine in {"strixvulkan", "strixvulkan075"}:
        excluded = ["fp8", "npu", "vision", "bf16", "reference"]
    else:
        excluded = ["npu", "vision", "reference"]
    return selected, [name for name in excluded if name not in selected]
