# Benchmarks and qualification

[Documentation index](README.md) · [Project README](../README.md)

Run commands from the repository root after installing the current checkout.
These procedures start real services and issue inference requests. For source-only
checks, use [Development](development.md).

- [Office profile matrix](#office-profile-matrix)
- [Long-context and reasoning benchmarks](#long-context-benchmark)
- [Raw result files](results/)

## Office profile matrix

For a long run under controlled office power, use the profile matrix. It runs
the source suite in its read-only, network-disabled Podman test container, then
starts, tests, benchmarks, and stops each ready profile sequentially. The first
run uses the checked-in same-host wall-time seeds; every later run sorts fastest
to slowest using successful elapsed times in `profile-history.tsv`:

```bash
# After installing the current checkout, run every ready profile.
tests/office-profile-matrix.sh --scope all

# Restrict a run to the active Qwen3.8 and DS4 optimization profiles.
tests/office-profile-matrix.sh --scope optimize
```

ROCmFPX profiles receive the exact-token, non-repeating 4K/32K benchmark with
three repetitions. Other LLM engines receive the fixed LongBench-v2 systems
sample; speech receives its modality smoke test. Results, environment snapshots,
per-profile logs, and ordering history are stored under
`/var/opt/halo-ai/state/benchmarks/office-profile-matrix`. Use `--smoke-only`
for a quick orchestration rehearsal. A cleanup trap stops managed inference
containers after normal completion, failure, or interruption.

The runner holds a transient systemd sleep inhibitor for the complete matrix
and independently detects any suspend interval from the boottime/monotonic
clock offset. Use `--exclude-profiles CSV` to preserve an explicit audit trail
when a suspected profile should not be retried unattended.

The runner selects `performance` with `powerprofilesctl` and requires both the
power-profiles daemon and ACPI platform profile to remain in that state before
and after every profile. Because the installed kernel does not expose a numeric
GPD WIN 5 DPTC limit, the runner does not equate the profile name with a watt
cap. Instead it samples the kernel's AMDGPU PPT sensor once per second and, for
a non-smoke run, requires the observed run peak to exceed 45 W. Override only
the observation threshold with `--minimum-ppt-watts N`; raw samples and the run
peak are retained with the results.

## Long-context benchmark

LongBench-v2 support uses a pinned dataset revision and SHA-256, the official
zero-shot prompt, exact chat-template-aware token counts from the active
llama.cpp backend, and resumable JSONL output. Start with the bounded canary:

```bash
halo-ai start qwen3.6-35b-a3b-q8xl-128k-lemonade
halo-ai bench longbench-v2 download
halo-ai bench longbench-v2 run qwen3.6-35b-a3b-q8xl-128k-lemonade
```

Then run the native-fit full subset. Inputs beyond 128K are reported as skipped,
not silently truncated:

```bash
halo-ai bench longbench-v2 run qwen3.6-35b-a3b-q8xl-128k-lemonade --suite full
```

For comparison with the upstream runner's middle-truncation policy, use a
separate output identity:

```bash
halo-ai bench longbench-v2 run qwen3.6-35b-a3b-q8xl-128k-lemonade \
  --suite full --overflow middle
```

Reasoning-policy comparisons use separate resumable outputs and preserve the
non-thinking default when the option is omitted:

```bash
halo-ai bench longbench-v2 run qwen3.8fp4 --sample-id SAMPLE \
  --max-tokens 1024 --reasoning-effort default
halo-ai bench longbench-v2 run qwen3.8fp4 --sample-id SAMPLE \
  --max-tokens 1024 --reasoning-effort medium
halo-ai bench longbench-v2 run qwen3.8fp4 --sample-id SAMPLE \
  --max-tokens 1024 --reasoning-effort xhigh
```

Every sample is flushed to disk, so rerunning the identical command resumes.
The score always reports completed, skipped, error, and truncated counts.

## Reasoning reliability

For Qwen3.8 final-answer reliability, run the paired,
interleaved structured-output suite. Two repetitions produce 24 matched calls per arm and
checkpoint atomically after every response:

```bash
halo-ai start qwen3.8fp4
halo-ai bench reasoning-reliability qwen3.8fp4 --repetitions 2 --max-tokens 1024
```

This separately reports API/protocol errors, empty content with `finish_reason`
`stop`, output-budget exhaustion, final-answer delivery, and validator results.
Its Wilson intervals and paired ratios support only a bounded claim for the
recorded host, runtime, suite, sampling policy, and output budget.

On the recorded FP4 host run, default-medium and xhigh each delivered and
validator-passed 24/24 structured calls (Wilson 95%: 86.2--100%), so the bounded
medium reliability gate passed while the superiority gate did not. A separate
literal ChatML-control-token sentinel failed empty-stop in both arms; raw
`<|im_start|>`/`<|im_end|>` content is therefore outside the qualified thinking
input boundary until it is escaped or rejected. See the
[machine summary](results/qwen3.8-reasoning-reliability-2026-08-30.json).

## NPU retrieval evaluation

Use an acquired [Halogen NPU profile](halogen-flash.md#npu-and-host-constraints).
Keep the SciFact dataset and embedding cache outside the checkout. The evaluator
verifies the archive SHA-256
`536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`;
quality mode also requires NumPy. Run stages sequentially without other clients:

```bash
mkdir -p /var/cache/halo-ai/benchmarks/scifact
curl --fail --location \
  https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip \
  --output /var/cache/halo-ai/benchmarks/scifact/scifact.zip
halo-ai start qwen3.8-halogen-v2-npu --switch
python3 tools/halogen_npu_validate.py --rounds 5 --output /tmp/npu-contention.json
python3 tools/halogen_npu_evaluate.py burst --output /tmp/npu-ingestion.json
python3 tools/halogen_npu_evaluate.py quality --output /tmp/npu-scifact.json
```

Quality mode embeds the full 5,183-document corpus and evaluates all 300 test
queries. It compares local BM25, embedding similarity, and top-ten reranking.
The resumable corpus cache is keyed by dataset hash, formatting, model,
dimensions, runtime version, and NPU artifact manifest. Queries and reranking
are recomputed; output includes per-query metrics and raw rankings.

Reranking the same top ten cannot improve recall@10, but can improve relevance
ordering (nDCG) and the first relevant result's position (MRR). Results support
only this dataset and pipeline, not the GPU generation model's quality. See
[Halogen qualification](halogen-flash.md#qualification-and-limitations) for
recorded results and their scope.

## Qualification policy

Pin dataset/evaluator versions, model hashes, and runtime identity. Keep prompts,
token budgets, reasoning policy, and cache conditions comparable. Report cold
and warm requests separately, include output lengths and total wall time, and
separate truncation, errors, task success, and speed. Confirm actual drafted and
accepted tokens before reporting speculative acceleration.

A smoke response establishes integration; a small fixed question set detects
regressions. Neither is a general coding-quality score. Configured context
capacity and successful allocation do not prove reliable retrieval at that
length. See the [Flash-Next procedure](qwen3.8-flash-test-plan.md) for a controlled
MTP matrix.

Broader qualification targets include executable coding suites such as
HumanEval+/MBPP+, multi-task long-context evaluation, and BF16-reference
perplexity/KL/top-1 agreement. These are separate evaluations, not implied by
the existing commands. Run generated code in an isolated evaluator. Record
planned work in the backlog and publish completed evidence under `docs/results/`.
