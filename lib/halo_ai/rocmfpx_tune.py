"""Pure same-host ROCmFPX tuning record helpers."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any


class TuneError(ValueError):
    """A malformed or incomparable tuning record."""


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


def summarize_context_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize repeated cold-cache ROCmFPX context runs by prompt length."""
    if not runs:
        raise TuneError("ROCmFPX benchmark has no runs")
    grouped: dict[int, list[dict[str, Any]]] = {}
    required = (
        "prompt_tokens", "ttft_seconds", "prompt_tokens_per_second",
        "decode_tokens_per_second", "peak_gtt_bytes", "peak_vram_bytes",
    )
    for index, run in enumerate(runs):
        if not isinstance(run, dict) or any(
            not isinstance(run.get(key), (int, float)) for key in required
        ):
            raise TuneError(f"ROCmFPX benchmark run {index} is missing numeric metrics")
        prompt_tokens = run["prompt_tokens"]
        if not isinstance(prompt_tokens, int) or prompt_tokens < 1:
            raise TuneError(f"ROCmFPX benchmark run {index} has invalid prompt tokens")
        grouped.setdefault(prompt_tokens, []).append(run)

    summaries = []
    for prompt_tokens, items in sorted(grouped.items()):
        drafted = sum(
            item["drafted_tokens"] for item in items
            if isinstance(item.get("drafted_tokens"), int)
        )
        accepted = sum(
            item["accepted_tokens"] for item in items
            if isinstance(item.get("accepted_tokens"), int)
        )
        summaries.append({
            "prompt_tokens": prompt_tokens,
            "repetitions": len(items),
            "ttft_seconds_median": round(statistics.median(
                item["ttft_seconds"] for item in items
            ), 6),
            "prompt_tokens_per_second_median": round(statistics.median(
                item["prompt_tokens_per_second"] for item in items
            ), 6),
            "decode_tokens_per_second_median": round(statistics.median(
                item["decode_tokens_per_second"] for item in items
            ), 6),
            "peak_gtt_bytes_max": max(item["peak_gtt_bytes"] for item in items),
            "peak_vram_bytes_max": max(item["peak_vram_bytes"] for item in items),
            "drafted_tokens": drafted or None,
            "accepted_tokens": accepted or None,
            "acceptance_percent": (
                round(accepted * 100 / drafted, 2) if drafted else None
            ),
        })
    return summaries


def compare_context_records(
    baseline: dict[str, Any], candidate: dict[str, Any],
) -> dict[str, Any]:
    """Compare exact-token ROCmFPX context records, including E2E crossover."""
    expected_kind = "halo-ai-rocmfpx-context-benchmark"
    if baseline.get("kind") != expected_kind or candidate.get("kind") != expected_kind:
        raise TuneError("context comparison requires ROCmFPX context benchmark records")
    baseline_runs = baseline.get("runs")
    candidate_runs = candidate.get("runs")
    if not isinstance(baseline_runs, list) or not isinstance(candidate_runs, list):
        raise TuneError("context comparison record has no runs")
    before_keys = {
        (item.get("prompt_tokens"), item.get("repetition")): item.get("prompt_sha256")
        for item in baseline_runs if isinstance(item, dict)
    }
    after_keys = {
        (item.get("prompt_tokens"), item.get("repetition")): item.get("prompt_sha256")
        for item in candidate_runs if isinstance(item, dict)
    }
    if not before_keys or before_keys != after_keys:
        raise TuneError("context records do not contain identical prompt-token runs")
    baseline_summary = {
        item.get("prompt_tokens"): item for item in baseline.get("summary", [])
        if isinstance(item, dict)
    }
    candidate_summary = {
        item.get("prompt_tokens"): item for item in candidate.get("summary", [])
        if isinstance(item, dict)
    }
    if not baseline_summary or baseline_summary.keys() != candidate_summary.keys():
        raise TuneError("context record summaries differ")
    completion_counts = {
        item.get("completion_tokens") for item in baseline_runs + candidate_runs
        if isinstance(item, dict)
    }
    if len(completion_counts) != 1 or not isinstance(next(iter(completion_counts)), int):
        raise TuneError("context records use different completion-token counts")
    completion_tokens = next(iter(completion_counts))

    rows = []
    for prompt_tokens in sorted(baseline_summary):
        before = baseline_summary[prompt_tokens]
        after = candidate_summary[prompt_tokens]
        required = (
            "ttft_seconds_median", "prompt_tokens_per_second_median",
            "decode_tokens_per_second_median", "peak_gtt_bytes_max",
        )
        if any(
            not isinstance(item.get(key), (int, float))
            for item in (before, after) for key in required
        ):
            raise TuneError(f"context {prompt_tokens} has incomplete summary metrics")
        baseline_tps = before["decode_tokens_per_second_median"]
        candidate_tps = after["decode_tokens_per_second_median"]
        prefill_delta = after["ttft_seconds_median"] - before["ttft_seconds_median"]
        decode_savings = 1 / baseline_tps - 1 / candidate_tps
        if prefill_delta <= 0 and decode_savings >= 0:
            crossover = 0
        elif prefill_delta > 0 and decode_savings > 0:
            crossover = math.ceil(prefill_delta / decode_savings)
        else:
            crossover = None
        baseline_e2e = before["ttft_seconds_median"] + completion_tokens / baseline_tps
        candidate_e2e = after["ttft_seconds_median"] + completion_tokens / candidate_tps
        rows.append({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "ttft_change_percent": _percent(
                after["ttft_seconds_median"], before["ttft_seconds_median"],
            ),
            "prompt_speed_change_percent": _percent(
                after["prompt_tokens_per_second_median"],
                before["prompt_tokens_per_second_median"],
            ),
            "decode_speed_change_percent": _percent(candidate_tps, baseline_tps),
            "peak_gtt_change_bytes": after["peak_gtt_bytes_max"] - before["peak_gtt_bytes_max"],
            "end_to_end_seconds_at_measured_completion": round(candidate_e2e, 6),
            "end_to_end_change_percent_at_measured_completion": _percent(
                candidate_e2e, baseline_e2e,
            ),
            "candidate_faster_after_generated_tokens": crossover,
            "candidate_acceptance_percent": after.get("acceptance_percent"),
        })
    return {
        "schema_version": 1,
        "kind": "halo-ai-rocmfpx-context-comparison",
        "baseline_profile": baseline.get("profile"),
        "candidate_profile": candidate.get("profile"),
        "identical_prompt_tokens": True,
        "contexts": rows,
    }
