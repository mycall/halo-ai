"""Pinned, dependency-free LongBench-v2 dataset and scoring helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable


DATASET_REPOSITORY = "zai-org/LongBench-v2"
DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
DATASET_FILENAME = "data.json"
DATASET_BYTES = 465_490_535
DATASET_SHA256 = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"
DATASET_SAMPLES = 503
DATASET_URL = (
    f"https://huggingface.co/datasets/{DATASET_REPOSITORY}/resolve/"
    f"{DATASET_REVISION}/{DATASET_FILENAME}"
)

PROMPT_TEMPLATE = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: \"The correct answer is (insert answer here)\"."""


class LongBenchError(RuntimeError):
    """A benchmark preparation or integrity failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_dataset(path: Path) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise LongBenchError(f"cannot inspect LongBench-v2 dataset {path}: {exc}") from exc
    if size != DATASET_BYTES:
        raise LongBenchError(
            f"LongBench-v2 size mismatch at {path}: expected {DATASET_BYTES}, got {size}"
        )
    digest = sha256_file(path)
    if digest != DATASET_SHA256:
        raise LongBenchError(
            f"LongBench-v2 SHA-256 mismatch at {path}: expected {DATASET_SHA256}, got {digest}"
        )


def download_dataset(destination: Path, progress: Callable[[int, int], None] | None = None) -> Path:
    """Download the pinned dataset once, then atomically publish it after verification."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_dataset(destination)
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > DATASET_BYTES:
        partial.unlink()
        offset = 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(DATASET_URL, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            # Some redirect targets ignore Range. Avoid appending a second full file.
            append = offset > 0 and getattr(response, "status", 200) == 206
            if not append:
                offset = 0
            with partial.open("ab" if append else "wb") as stream:
                copied = offset
                while True:
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
                    copied += len(block)
                    if progress:
                        progress(copied, DATASET_BYTES)
                stream.flush()
                os.fsync(stream.fileno())
    except OSError as exc:
        raise LongBenchError(f"LongBench-v2 download failed; partial file retained at {partial}: {exc}") from exc
    verify_dataset(partial)
    os.replace(partial, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return destination


def load_dataset(path: Path, *, require_pinned: bool = True) -> list[dict[str, Any]]:
    if require_pinned:
        verify_dataset(path)
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise LongBenchError(f"cannot load LongBench-v2 dataset {path}: {exc}") from exc
    if not isinstance(value, list):
        raise LongBenchError("LongBench-v2 root must be a JSON array")
    required = {
        "_id", "domain", "sub_domain", "difficulty", "length", "question",
        "choice_A", "choice_B", "choice_C", "choice_D", "answer", "context",
    }
    for index, item in enumerate(value):
        if not isinstance(item, dict) or not required.issubset(item):
            raise LongBenchError(f"LongBench-v2 sample {index} has an unexpected schema")
        if item["answer"] not in {"A", "B", "C", "D"}:
            raise LongBenchError(f"LongBench-v2 sample {item['_id']} has an invalid answer")
    if require_pinned and len(value) != DATASET_SAMPLES:
        raise LongBenchError(
            f"LongBench-v2 sample count mismatch: expected {DATASET_SAMPLES}, got {len(value)}"
        )
    return value


def render_prompt(item: dict[str, Any], context: str | None = None) -> str:
    replacements = {
        "$DOC$": item["context"] if context is None else context,
        "$Q$": item["question"],
        "$C_A$": item["choice_A"],
        "$C_B$": item["choice_B"],
        "$C_C$": item["choice_C"],
        "$C_D$": item["choice_D"],
    }
    prompt = PROMPT_TEMPLATE
    for marker, value in replacements.items():
        prompt = prompt.replace(marker, str(value).strip())
    return prompt


def extract_answer(response: str) -> str | None:
    """Use the official evaluator's two accepted answer forms."""
    plain = response.replace("*", "")
    for pattern in (
        r"The correct answer is \(([A-D])\)",
        r"The correct answer is ([A-D])",
    ):
        match = re.search(pattern, plain)
        if match:
            return match.group(1)
    return None


def select_items(
    items: Iterable[dict[str, Any]], *, suite: str, limit: int | None = None,
    length: str | None = None, difficulty: str | None = None, domain: str | None = None,
    sample_id: str | None = None,
) -> list[dict[str, Any]]:
    selected = [
        item for item in items
        if (length is None or item["length"] == length)
        and (difficulty is None or item["difficulty"] == difficulty)
        and (domain is None or item["domain"] == domain)
        and (sample_id is None or str(item["_id"]) == sample_id)
    ]
    if sample_id is not None and len(selected) != 1:
        raise LongBenchError(f"LongBench-v2 sample ID not found under active filters: {sample_id}")
    if suite == "canary" and sample_id is None:
        # One stable sample from every available difficulty/length cell.
        cells: dict[tuple[str, str], dict[str, Any]] = {}
        for item in sorted(selected, key=lambda value: str(value["_id"])):
            cells.setdefault((item["difficulty"], item["length"]), item)
        selected = [cells[key] for key in sorted(cells)]
    if limit is not None:
        selected = selected[:limit]
    return selected


def middle_context(context: str, keep_chars: int) -> str:
    if keep_chars >= len(context):
        return context
    if keep_chars <= 0:
        return ""
    left = keep_chars // 2
    right = keep_chars - left
    return context[:left] + "\n\n[... middle truncated by halo-ai ...]\n\n" + context[-right:]


def truncate_to_budget(
    item: dict[str, Any], budget: int, count_tokens: Callable[[str], int],
) -> tuple[str, int]:
    """Middle-truncate only the document, measuring the actual rendered chat request."""
    context = str(item["context"])
    full_prompt = render_prompt(item)
    full_count = count_tokens(full_prompt)
    if full_count <= budget:
        return full_prompt, full_count
    empty_prompt = render_prompt(item, "")
    empty_count = count_tokens(empty_prompt)
    if empty_count > budget:
        raise LongBenchError("question and choices alone exceed the profile token budget")
    low, high = 0, len(context)
    best_prompt, best_count = empty_prompt, empty_count
    while low <= high:
        midpoint = (low + high) // 2
        candidate = render_prompt(item, middle_context(context, midpoint))
        count = count_tokens(candidate)
        if count <= budget:
            best_prompt, best_count = candidate, count
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best_prompt, best_count


def score_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    completed = [item for item in rows if item.get("status") == "complete"]
    skipped = [item for item in rows if item.get("status") == "skipped_overflow"]
    errors = [item for item in rows if item.get("status") == "error"]

    def group(field: str) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for name in sorted({str(item[field]) for item in completed}):
            subset = [item for item in completed if str(item[field]) == name]
            correct = sum(bool(item.get("judge")) for item in subset)
            result[name] = {
                "samples": len(subset), "correct": correct,
                "accuracy_percent": round(100 * correct / len(subset), 1),
            }
        return result

    correct = sum(bool(item.get("judge")) for item in completed)
    return {
        "records": len(rows),
        "completed": len(completed),
        "skipped_overflow": len(skipped),
        "errors": len(errors),
        "truncated": sum(bool(item.get("truncated")) for item in completed),
        "correct": correct,
        "accuracy_percent": round(100 * correct / len(completed), 1) if completed else None,
        "by_difficulty": group("difficulty"),
        "by_length": group("length"),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("record is not an object")
                    rows.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise LongBenchError(f"cannot read benchmark results {path}: {exc}") from exc
    return rows


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
