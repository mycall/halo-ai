"""Pure scoring helpers for ROCmFPX and speculative-draft experiments."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


class TuneError(ValueError):
    """A malformed or incomparable tuning record."""


PROPOSAL_LENGTHS = (1, 2, 4, 6)
REQUIRED_STAGE3_DOMAINS = {
    "chat", "code", "json-tool", "multilingual", "reasoning",
}
REQUIRED_STAGE3_MODES = {"thinking", "nonthinking"}


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TuneError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TuneError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TuneError(f"tuning record must be a JSON object: {path}")
    return value


def _percent(candidate: float, baseline: float) -> float:
    if baseline == 0:
        raise TuneError("cannot compare a zero baseline metric")
    return round((candidate / baseline - 1) * 100, 2)


def compare_mtp_records(
    baseline: dict[str, Any], candidate: dict[str, Any],
) -> dict[str, Any]:
    """Compare same-host Halo ROCmFPX baseline and MTP machine records."""
    baseline_runs = baseline.get("server_benchmarks")
    candidate_runs = candidate.get("benchmarks")
    if not isinstance(baseline_runs, list) or not baseline_runs:
        raise TuneError("baseline record has no server_benchmarks")
    if not isinstance(candidate_runs, list) or not candidate_runs:
        raise TuneError("candidate record has no benchmarks")
    baseline_by_prompt = {item.get("prompt_tokens"): item for item in baseline_runs}
    candidate_by_prompt = {item.get("prompt_tokens"): item for item in candidate_runs}
    if baseline_by_prompt.keys() != candidate_by_prompt.keys():
        raise TuneError("baseline and candidate prompt-token sets differ")

    baseline_sha = (baseline.get("model") or {}).get("sha256")
    candidate_sha = (candidate.get("reused_model") or {}).get("sha256")
    same_model = bool(baseline_sha) and baseline_sha == candidate_sha
    rows = []
    for prompt_tokens in sorted(baseline_by_prompt):
        before = baseline_by_prompt[prompt_tokens]
        after = candidate_by_prompt[prompt_tokens]
        required = (
            "ttft_seconds", "prompt_tokens_per_second",
            "decode_tokens_per_second", "peak_gtt_bytes",
        )
        if any(not isinstance(item.get(key), (int, float)) for item in (before, after) for key in required):
            raise TuneError(f"context {prompt_tokens} is missing a numeric benchmark metric")
        row = {
            "prompt_tokens": prompt_tokens,
            "ttft_change_percent": _percent(after["ttft_seconds"], before["ttft_seconds"]),
            "prompt_speed_change_percent": _percent(
                after["prompt_tokens_per_second"], before["prompt_tokens_per_second"],
            ),
            "decode_speed_change_percent": _percent(
                after["decode_tokens_per_second"], before["decode_tokens_per_second"],
            ),
            "peak_gtt_change_bytes": after["peak_gtt_bytes"] - before["peak_gtt_bytes"],
            "drafted_tokens": after.get("drafted_tokens"),
            "accepted_tokens": after.get("accepted_tokens"),
            "acceptance_percent": after.get("acceptance_percent"),
        }
        rows.append(row)

    conformance = candidate.get("conformance") or {}
    lifecycle = candidate.get("lifecycle") or {}
    same_prompts = (candidate.get("benchmark_method") or {}).get("same_prompts") is True
    lifecycle_pass = all(
        lifecycle.get(key) == "pass"
        for key in ("start", "readiness", "deterministic_smoke", "stop")
    ) and lifecycle.get("oom_or_device_reset") is False
    semantic_pass = (
        isinstance(conformance.get("corpus_cases"), int)
        and conformance.get("semantic_matches") == conformance.get("corpus_cases")
    )
    strict_identity = conformance.get("strict_token_identity") == "proven"
    materially_faster = all(row["decode_speed_change_percent"] >= 10 for row in rows)
    evidence = {
        "same_model_sha256": same_model,
        "same_prompts": same_prompts,
        "same_host_evidence_only": True,
        "lifecycle_pass": lifecycle_pass,
        "semantic_corpus_pass": semantic_pass,
        "strict_token_identity_proven": strict_identity,
        "decode_at_least_10_percent_faster_every_context": materially_faster,
    }
    if not all((same_model, same_prompts, lifecycle_pass, semantic_pass)):
        decision = "fail"
        reason = "candidate evidence is not comparable, stable, and semantically conformant"
    elif not strict_identity:
        decision = "hold-experimental"
        reason = "strict cross-process token identity remains unproven"
    elif not materially_faster:
        decision = "retain-baseline"
        reason = "candidate does not improve decode by at least 10% at every measured context"
    else:
        decision = "pass"
        reason = "strict conformance and the material decode-speed gate pass"
    return {
        "schema_version": 1,
        "kind": "halo-ai-rocmfpx-mtp-comparison",
        "baseline_profile": baseline.get("profile"),
        "candidate_profile": candidate.get("profile"),
        "model_sha256": baseline_sha,
        "contexts": rows,
        "conformance": conformance,
        "evidence": evidence,
        "decision": decision,
        "reason": reason,
    }


def token_sha256(tokens: Iterable[int]) -> str:
    canonical = json.dumps(list(tokens), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[rank])


def _score_group(records: list[dict[str, Any]], proposal_length: int) -> dict[str, Any]:
    accepted_runs: list[int] = []
    first_token_matches = 0
    for record in records:
        target = record["target_tokens"]
        draft = record["draft_tokens"][:proposal_length]
        accepted = 0
        for expected, proposed in zip(target, draft):
            if expected != proposed:
                break
            accepted += 1
        accepted_runs.append(accepted)
        first_token_matches += int(bool(draft) and bool(target) and draft[0] == target[0])
    count = len(accepted_runs)
    return {
        "proposals": count,
        "proposal_length": proposal_length,
        "first_token_agreement_percent": round(first_token_matches * 100 / count, 2),
        "zero_acceptance_percent": round(sum(value == 0 for value in accepted_runs) * 100 / count, 2),
        "mean_accepted_tokens": round(statistics.fmean(accepted_runs), 3),
        "accepted_run_p50": _percentile(accepted_runs, 0.50),
        "accepted_run_p95": _percentile(accepted_runs, 0.95),
        "full_proposal_acceptance_percent": round(
            sum(value == proposal_length for value in accepted_runs) * 100 / count, 2,
        ),
    }


def score_stage3_records(document: dict[str, Any]) -> dict[str, Any]:
    """Score token proposals captured from exact target-rendered prefixes."""
    if document.get("schema_version") != 1:
        raise TuneError("Stage 3 record schema_version must be 1")
    if document.get("kind") != "halo-ai-stage3-version-mix":
        raise TuneError("not a halo-ai Stage 3 version-mix record")
    if document.get("target_model_family") != "Qwen3.8":
        raise TuneError("the sole Stage 3 verifier must be Qwen3.8")
    if document.get("draft_model_family") != "Qwen3.6":
        raise TuneError("this no-download proxy scorer is restricted to Qwen3.6")
    if document.get("target_tokenizer_owner") != "Qwen3.8":
        raise TuneError("prompts must be rendered and tokenized by Qwen3.8")
    records = document.get("proposals")
    if not isinstance(records, list) or not records:
        raise TuneError("Stage 3 record contains no proposals")

    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TuneError(f"proposal {index} is not an object")
        target = record.get("target_tokens")
        draft = record.get("draft_tokens")
        if (
            not isinstance(target, list) or not target
            or not isinstance(draft, list) or len(draft) < max(PROPOSAL_LENGTHS)
            or any(not isinstance(token, int) or token < 0 for token in target + draft)
        ):
            raise TuneError(f"proposal {index} has invalid target/draft token IDs")
        target_hash = record.get("target_prefix_sha256")
        draft_hash = record.get("draft_prefix_sha256")
        if (
            not isinstance(target_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", target_hash) is None
            or target_hash != draft_hash
        ):
            raise TuneError(f"proposal {index} was not produced from an identical token prefix")
        domain = record.get("domain")
        mode = record.get("mode")
        bucket = record.get("context_bucket")
        if domain not in REQUIRED_STAGE3_DOMAINS:
            raise TuneError(f"proposal {index} has unsupported domain {domain!r}")
        if mode not in REQUIRED_STAGE3_MODES:
            raise TuneError(f"proposal {index} has unsupported mode {mode!r}")
        if bucket not in {"short", "4k", "32k"}:
            raise TuneError(f"proposal {index} has unsupported context bucket {bucket!r}")
        normalized.append(record)

    domains = {record["domain"] for record in normalized}
    modes = {record["mode"] for record in normalized}
    missing_domains = sorted(REQUIRED_STAGE3_DOMAINS - domains)
    missing_modes = sorted(REQUIRED_STAGE3_MODES - modes)
    grouped: dict[str, Any] = {}
    dimensions = {
        "aggregate": {"all": normalized},
        "domain": {value: [item for item in normalized if item["domain"] == value] for value in sorted(domains)},
        "domain_mode": {
            f"{domain}/{mode}": [
                item for item in normalized
                if item["domain"] == domain and item["mode"] == mode
            ]
            for domain in sorted(domains)
            for mode in sorted(modes)
        },
        "context_bucket": {value: [item for item in normalized if item["context_bucket"] == value] for value in sorted({item["context_bucket"] for item in normalized})},
        "mode": {value: [item for item in normalized if item["mode"] == value] for value in sorted(modes)},
    }
    for dimension, groups in dimensions.items():
        grouped[dimension] = {
            name: {str(length): _score_group(items, length) for length in PROPOSAL_LENGTHS}
            for name, items in groups.items()
        }
    corpus_complete = not missing_domains and not missing_modes and {
        "short", "4k", "32k",
    } <= {record["context_bucket"] for record in normalized}
    mode_one = {
        mode: grouped["mode"][mode]["1"]
        for mode in sorted(modes)
    }
    failing_modes = [
        mode for mode, metrics in mode_one.items()
        if metrics["first_token_agreement_percent"] < 50
        or metrics["zero_acceptance_percent"] > 50
    ]
    promising_modes = [
        mode for mode in sorted(modes)
        if grouped["mode"][mode]["1"]["first_token_agreement_percent"] >= 80
        and grouped["mode"][mode]["6"]["mean_accepted_tokens"] >= 3
    ]
    if failing_modes:
        decision = "do-not-expand-general-proxy"
    elif corpus_complete:
        decision = "measure-latency"
    else:
        decision = "incomplete-corpus"
    return {
        "schema_version": 1,
        "kind": "halo-ai-stage3-version-mix-score",
        "target_model": document.get("target_model"),
        "draft_model": document.get("draft_model"),
        "proposal_count": len(normalized),
        "prefix_identity_verified": True,
        "corpus_complete": corpus_complete,
        "missing_domains": missing_domains,
        "missing_modes": missing_modes,
        "metrics": grouped,
        "screen_gate": {
            "minimum_first_token_agreement_percent_per_mode": 50,
            "maximum_zero_acceptance_percent_per_mode": 50,
            "promising_mode_first_token_agreement_percent": 80,
            "promising_mode_mean_accepted_tokens_at_6": 3,
            "failing_modes": failing_modes,
            "promising_modes": promising_modes,
        },
        "download_authorized": False,
        "decision": decision,
    }
