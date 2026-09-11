"""Fixed, download-free ROCmFPX quality and strict-identity helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


class QualityError(ValueError):
    """A malformed suite, result, or comparison."""


def load_suite(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        suite = json.loads(raw)
    except OSError as exc:
        raise QualityError(f"cannot read quality suite {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise QualityError(f"invalid quality suite JSON {path}: {exc}") from exc
    if not isinstance(suite, dict) or suite.get("schema_version") != 1:
        raise QualityError("quality suite must be a schema-version 1 object")
    if not isinstance(suite.get("suite_id"), str) or not suite["suite_id"]:
        raise QualityError("quality suite has no suite_id")
    system = suite.get("system_prompt")
    cases = suite.get("cases")
    if not isinstance(system, str) or not system or not isinstance(cases, list) or not cases:
        raise QualityError("quality suite requires a system prompt and cases")
    identifiers: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise QualityError(f"quality case {index} has no ID")
        if case["id"] in identifiers:
            raise QualityError(f"duplicate quality case {case['id']}")
        identifiers.add(case["id"])
        if not isinstance(case.get("category"), str):
            raise QualityError(f"quality case {case['id']} has no category")
        if not isinstance(case.get("max_tokens"), int) or not 1 <= case["max_tokens"] <= 512:
            raise QualityError(f"quality case {case['id']} has invalid max_tokens")
        validator = case.get("validator")
        if not isinstance(validator, dict) or validator.get("type") not in {"exact", "json"}:
            raise QualityError(f"quality case {case['id']} has an invalid validator")
        if "expected" not in validator:
            raise QualityError(f"quality case {case['id']} has no expected value")
        sources = sum(key in case for key in ("prompt", "messages", "builder"))
        if sources != 1:
            raise QualityError(f"quality case {case['id']} must define one prompt source")
    return suite, hashlib.sha256(raw).hexdigest()


def case_messages(suite: dict[str, Any], case: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": suite["system_prompt"]},
    ]
    if "prompt" in case:
        messages.append({"role": "user", "content": case["prompt"]})
    elif "messages" in case:
        supplied = case["messages"]
        if not isinstance(supplied, list) or not supplied:
            raise QualityError(f"quality case {case['id']} has invalid messages")
        messages.extend(copy.deepcopy(supplied))
    else:
        builder = case["builder"]
        if (
            not isinstance(builder, dict)
            or builder.get("type") not in {"record_needle", "record_needle_json"}
        ):
            raise QualityError(f"quality case {case['id']} has an unsupported builder")
        records = builder.get("records")
        target = builder.get("target_index")
        expected = builder.get("expected")
        if (
            not isinstance(records, int) or not 32 <= records <= 2048
            or not isinstance(target, int) or not 0 <= target < records
            or not isinstance(expected, str) or not expected
        ):
            raise QualityError(f"quality case {case['id']} has invalid builder values")
        lines = []
        for index in range(records):
            code = expected if index == target else f"delta-{(index * 7919 + 104729) % 100000:05d}"
            lines.append(
                f"Archive record {index:06d}: owner=team-{index % 97:02d}; "
                f"state=closed; retrieval_code={code}."
            )
        if builder["type"] == "record_needle_json":
            lines.append(
                "Return only one JSON object with exactly the keys record and "
                f"retrieval_code for archive record {target:06d}. The record value "
                "must be an integer."
            )
        else:
            lines.append(
                f"Return only the retrieval_code from archive record {target:06d}."
            )
        messages.append({"role": "user", "content": "\n".join(lines)})
    for item in messages:
        if (
            not isinstance(item, dict) or item.get("role") not in {"system", "user", "assistant"}
            or not isinstance(item.get("content"), str)
        ):
            raise QualityError(f"quality case {case['id']} has malformed messages")
    return messages


def score_case(case: dict[str, Any], content: str) -> tuple[bool, str]:
    validator = case["validator"]
    normalized = content.strip()
    if validator["type"] == "exact":
        expected = validator["expected"]
        if not isinstance(expected, str):
            raise QualityError(f"quality case {case['id']} exact expected value is not text")
        return normalized == expected, normalized
    candidate = normalized
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return False, normalized
    return parsed == validator["expected"], json.dumps(
        parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )


def wilson_interval(passed: int, total: int) -> list[float]:
    if total < 1 or not 0 <= passed <= total:
        raise QualityError("invalid Wilson interval counts")
    z = 1.959963984540054
    proportion = passed / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [round(max(0.0, centre - margin) * 100, 2), round(min(1.0, centre + margin) * 100, 2)]


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise QualityError("quality benchmark has no results")
    passed = sum(item.get("passed") is True for item in results)
    categories: dict[str, dict[str, int]] = {}
    for item in results:
        category = item.get("category")
        if not isinstance(category, str):
            raise QualityError("quality benchmark result has no category")
        row = categories.setdefault(category, {"passed": 0, "total": 0})
        row["total"] += 1
        row["passed"] += item.get("passed") is True
    return {
        "passed": passed,
        "total": len(results),
        "score_percent": round(passed * 100 / len(results), 2),
        "wilson_95_percent": wilson_interval(passed, len(results)),
        "categories": categories,
    }


def _median(values: list[float | int]) -> float | None:
    return round(float(statistics.median(values)), 6) if values else None


def _rate_summary(successes: int, total: int) -> dict[str, Any]:
    return {
        "count": successes,
        "percent": round(successes * 100 / total, 2),
        "wilson_95_percent": wilson_interval(successes, total),
    }


def summarize_reasoning_reliability(
    records: list[dict[str, Any]], *, minimum_calls_per_arm: int = 24,
) -> dict[str, Any]:
    """Summarize paired final-answer delivery without overstating model quality."""
    if not records:
        raise QualityError("reasoning reliability benchmark has no records")
    if minimum_calls_per_arm < 1:
        raise QualityError("minimum calls per arm must be positive")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        arm = record.get("arm")
        if arm not in {"default", "xhigh"}:
            raise QualityError("reasoning reliability record has an invalid arm")
        grouped.setdefault(arm, []).append(record)

    arms: dict[str, dict[str, Any]] = {}
    for arm in ("default", "xhigh"):
        rows = grouped.get(arm, [])
        if not rows:
            raise QualityError(f"reasoning reliability benchmark has no {arm} records")
        total = len(rows)
        request_errors = sum(row.get("status") == "request_error" for row in rows)
        invalid_responses = sum(row.get("status") == "invalid_response" for row in rows)
        final_present = sum(row.get("final_content_present") is True for row in rows)
        empty_stop = sum(row.get("empty_stop_failure") is True for row in rows)
        budget_exhaustions = sum(row.get("output_budget_exhausted") is True for row in rows)
        delivery_successes = sum(row.get("delivery_success") is True for row in rows)
        validator_passes = sum(row.get("validator_passed") is True for row in rows)
        completion_tokens = [
            value for row in rows
            if isinstance((value := row.get("completion_tokens")), int) and value >= 0
        ]
        wall_seconds = [
            value for row in rows
            if isinstance((value := row.get("wall_seconds")), (int, float)) and value >= 0
        ]
        decode_rates = [
            value for row in rows
            if isinstance((value := row.get("decode_tokens_per_second")), (int, float))
            and value > 0
        ]
        arms[arm] = {
            "calls": total,
            "request_errors": request_errors,
            "invalid_responses": invalid_responses,
            "final_content": _rate_summary(final_present, total),
            "empty_stop_failures": _rate_summary(empty_stop, total),
            "output_budget_exhaustions": _rate_summary(budget_exhaustions, total),
            "delivery_success": _rate_summary(delivery_successes, total),
            "validator_pass": _rate_summary(validator_passes, total),
            "median_completion_tokens": _median(completion_tokens),
            "median_wall_seconds": _median(wall_seconds),
            "median_decode_tokens_per_second": _median(decode_rates),
        }

    paired: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for record in records:
        repetition = record.get("repetition")
        case_id = record.get("case_id")
        if not isinstance(repetition, int) or not isinstance(case_id, str):
            raise QualityError("reasoning reliability record has an invalid pair identity")
        pair = paired.setdefault((repetition, case_id), {})
        arm = record["arm"]
        if arm in pair:
            raise QualityError(f"duplicate reasoning reliability record for {repetition}/{case_id}/{arm}")
        pair[arm] = record

    complete_pairs = [pair for pair in paired.values() if set(pair) == {"default", "xhigh"}]
    default_only_failure = sum(
        pair["default"].get("delivery_success") is not True
        and pair["xhigh"].get("delivery_success") is True
        for pair in complete_pairs
    )
    xhigh_only_failure = sum(
        pair["default"].get("delivery_success") is True
        and pair["xhigh"].get("delivery_success") is not True
        for pair in complete_pairs
    )
    both_failure = sum(
        pair["default"].get("delivery_success") is not True
        and pair["xhigh"].get("delivery_success") is not True
        for pair in complete_pairs
    )
    neither_failure = len(complete_pairs) - default_only_failure - xhigh_only_failure - both_failure
    discordant = default_only_failure + xhigh_only_failure
    mcnemar_p = None
    if discordant:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(default_only_failure, xhigh_only_failure) + 1)
        ) / (2 ** discordant)
        mcnemar_p = round(min(1.0, 2 * tail), 8)

    def paired_ratios(field: str) -> list[float]:
        ratios = []
        for pair in complete_pairs:
            left = pair["default"].get(field)
            right = pair["xhigh"].get(field)
            if (
                isinstance(left, (int, float)) and left >= 0
                and isinstance(right, (int, float)) and right > 0
            ):
                ratios.append(left / right)
        return ratios

    default_summary = arms["default"]
    enough_calls = all(arms[arm]["calls"] >= minimum_calls_per_arm for arm in arms)
    bounded_reliability_supported = (
        enough_calls
        and default_summary["request_errors"] == 0
        and default_summary["invalid_responses"] == 0
        and default_summary["delivery_success"]["count"] == default_summary["calls"]
    )
    comparative_superiority_supported = (
        bounded_reliability_supported
        and xhigh_only_failure > default_only_failure
        and mcnemar_p is not None and mcnemar_p < 0.05
    )
    operational_rows = [
        row for row in records if row.get("category") != "special_token_boundary"
    ]
    operational_arms: dict[str, dict[str, Any]] = {}
    for arm in ("default", "xhigh"):
        rows = [row for row in operational_rows if row["arm"] == arm]
        total = len(rows)
        delivered = sum(row.get("delivery_success") is True for row in rows)
        empty_stop = sum(row.get("empty_stop_failure") is True for row in rows)
        operational_arms[arm] = {
            "calls": total,
            "delivery_success": _rate_summary(delivered, total) if total else None,
            "empty_stop_failures": _rate_summary(empty_stop, total) if total else None,
            "validator_passes": sum(row.get("validator_passed") is True for row in rows),
        }
    operational_minimum_met = all(
        operational_arms[arm]["calls"] >= minimum_calls_per_arm for arm in operational_arms
    )
    operational_default_supported = (
        operational_minimum_met
        and operational_arms["default"]["delivery_success"]["count"]
        == operational_arms["default"]["calls"]
    )
    boundary_rows = [
        row for row in records if row.get("category") == "special_token_boundary"
    ]
    return {
        "calls": len(records),
        "minimum_calls_per_arm": minimum_calls_per_arm,
        "arms": arms,
        "paired_delivery": {
            "complete_pairs": len(complete_pairs),
            "neither_failed": neither_failure,
            "both_failed": both_failure,
            "default_only_failed": default_only_failure,
            "xhigh_only_failed": xhigh_only_failure,
            "exact_mcnemar_two_sided_p": mcnemar_p,
        },
        "paired_median_default_to_xhigh_ratios": {
            "completion_tokens": _median(paired_ratios("completion_tokens")),
            "wall_seconds": _median(paired_ratios("wall_seconds")),
            "decode_tokens_per_second": _median(paired_ratios("decode_tokens_per_second")),
        },
        "cohorts": {
            "ordinary_non_control_token": {
                "excluded_category": "special_token_boundary",
                "arms": operational_arms,
            },
            "special_token_boundary": {
                "calls": len(boundary_rows),
                "delivery_failures": sum(
                    row.get("delivery_success") is not True for row in boundary_rows
                ),
                "empty_stop_failures": sum(
                    row.get("empty_stop_failure") is True for row in boundary_rows
                ),
            },
        },
        "claim_gate": {
            "minimum_sample_met": enough_calls,
            "bounded_default_reliability_supported": bounded_reliability_supported,
            "ordinary_minimum_sample_met": operational_minimum_met,
            "bounded_default_ordinary_reliability_supported": operational_default_supported,
            "ordinary_scope_exclusion": "literal chat-control-token boundary prompts",
            "comparative_superiority_supported": comparative_superiority_supported,
            "scope": "this host, profile, suite, runtime, sampling policy, and output budget only",
        },
    }


def compare_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) < 2:
        raise QualityError("quality comparison requires at least two records")
    expected_kind = "halo-ai-rocmfpx-quality-benchmark"
    if any(record.get("kind") != expected_kind for record in records):
        raise QualityError("quality comparison received a non-quality record")
    suite_hashes = {record.get("suite_sha256") for record in records}
    if len(suite_hashes) != 1 or not next(iter(suite_hashes)):
        raise QualityError("quality records use different suites")
    case_sets = [
        tuple(item.get("case_id") for item in record.get("results", []))
        for record in records
    ]
    if not case_sets[0] or any(value != case_sets[0] for value in case_sets[1:]):
        raise QualityError("quality records use different ordered case sets")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        profile = record.get("profile")
        if not isinstance(profile, str):
            raise QualityError("quality record has no profile")
        grouped.setdefault(profile, []).append(record)

    profiles: dict[str, dict[str, Any]] = {}
    for profile, items in grouped.items():
        repetitions = len(items)
        per_case: dict[str, list[dict[str, Any]]] = {
            case_id: [] for case_id in case_sets[0]
        }
        for item in items:
            for result in item["results"]:
                per_case[result["case_id"]].append(result)
        consistent_cases = sum(
            len({row.get("output_token_sha256") for row in rows}) == 1
            for rows in per_case.values()
        )
        case_passes = sum(all(row.get("passed") is True for row in rows) for rows in per_case.values())
        profiles[profile] = {
            "process_repetitions": repetitions,
            "model_sha256": (items[0].get("model") or {}).get("sha256"),
            "quantization": (items[0].get("model") or {}).get("quantization"),
            "features": items[0].get("features", []),
            "cases_passed_every_repetition": case_passes,
            "cases_total": len(per_case),
            "score_percent": round(case_passes * 100 / len(per_case), 2),
            "wilson_95_percent": wilson_interval(case_passes, len(per_case)),
            "self_consistent_cases": consistent_cases,
            "self_consistent": repetitions >= 2 and consistent_cases == len(per_case),
        }

    strict_identity: dict[str, dict[str, Any]] = {}
    by_model: dict[str, list[str]] = {}
    for profile, summary in profiles.items():
        sha = summary["model_sha256"]
        if isinstance(sha, str):
            by_model.setdefault(sha, []).append(profile)
    for same_model_profiles in by_model.values():
        baselines = [
            profile for profile in same_model_profiles
            if not {"mtp", "dflash"}.intersection(profiles[profile]["features"])
        ]
        candidates = [
            profile for profile in same_model_profiles
            if {"mtp", "dflash"}.intersection(profiles[profile]["features"])
        ]
        if len(baselines) != 1:
            continue
        baseline = baselines[0]
        baseline_rows = {
            row["case_id"]: row["output_token_sha256"]
            for row in grouped[baseline][0]["results"]
        }
        for candidate in candidates:
            candidate_rows = {
                row["case_id"]: row["output_token_sha256"]
                for row in grouped[candidate][0]["results"]
            }
            matching = sum(
                candidate_rows.get(case_id) == token_hash
                for case_id, token_hash in baseline_rows.items()
            )
            proven = (
                profiles[baseline]["self_consistent"]
                and profiles[candidate]["self_consistent"]
                and matching == len(baseline_rows)
            )
            strict_identity[candidate] = {
                "baseline_profile": baseline,
                "matching_cases": matching,
                "cases_total": len(baseline_rows),
                "status": "proven" if proven else "not-proven",
            }

    fp4_baseline = profiles.get("qwen3.8-27b-rocmfp4-baseline")
    fp8_baseline = profiles.get("qwen3.8-27b-rocmfp8-baseline")
    fp8_vs_fp4 = None
    if fp4_baseline and fp8_baseline:
        fp8_vs_fp4 = {
            "fp4_score_percent": fp4_baseline["score_percent"],
            "fp8_score_percent": fp8_baseline["score_percent"],
            "difference_percentage_points": round(
                fp8_baseline["score_percent"] - fp4_baseline["score_percent"], 2,
            ),
            "interpretation": "This bounded suite can detect a difference but cannot prove equivalence.",
        }
    return {
        "schema_version": 1,
        "kind": "halo-ai-rocmfpx-quality-comparison",
        "suite_sha256": next(iter(suite_hashes)),
        "profiles": profiles,
        "strict_identity": strict_identity,
        "fp8_vs_fp4": fp8_vs_fp4,
    }
