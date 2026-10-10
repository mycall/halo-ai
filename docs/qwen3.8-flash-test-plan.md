# Flash-Next GGUF qualification

[Reference manual](README.md) · [Qwen runtimes](qwen-profiles.md) · [Benchmarks](benchmarking.md)

This procedure qualifies the UD-Q4_K_XL target and its shared MTP heads. It
covers standalone build 10715 and Strix Vulkan v0.7.5. Halogen's native HGN
checkpoint has a separate [validation workflow](halogen-flash.md#validation-commands-and-evidence).

## Prerequisites

Use the [Qwen artifact and runtime settings](qwen-profiles.md#qwen38-flash-next-ud-q4_k_xl-and-mtp).
Acquire the selected profile, verify its model files with `models verify --full`,
and inspect `profiles render` before loading. Keep context at 32,768, use one
slot and Q8 K/V, and disable automatic fitting. The shared head must load as a
draft of its matching target; it is not an independent model.

The ordinary build-10715 MTP profile uses width 2 and batch/ubatch 8192/2048.
Its rendering must include:

```text
--spec-type draft-mtp
--spec-draft-model /models/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
--spec-draft-ngl all
--spec-draft-n-max 2
--fit off
```

A cold load may return HTTP 503 `Loading model` within the 1,200-second startup
allowance. `/v1/models` proves readiness only; a smoke test must also observe
nonzero drafted and accepted tokens.

## Controlled comparison

1. Establish target-only and width-2 baselines using identical greedy code and
   prose prompts, token budgets, reasoning settings, and cache policy.
2. Repeat each arm in at least two fresh processes. Record runtime identity,
   model hashes, cold-load time, peak GTT/RSS, host available memory, prefill,
   decode, wall time, draft proposals/acceptance, outputs, and kernel errors.
3. Change one variable at a time: widths 3 and 4, then the shared Q8_0 head.
   Keep target, context, and other settings fixed.
4. Compare normalized output and token-stream identity as well as task scores.
   Equal scores can conceal different incorrect answers.
5. Increase context only after repeated clean load, inference, and stop cycles.
   Follow [memory-failure recovery](operations.md#memory-failures-and-recovery)
   if a trial fails.

A candidate needs better end-to-end time on the intended workload, nonzero
speculation, sufficient memory reserve, and matching greedy outputs before
promotion as an equivalent accelerated default. Acceptance percentage or decode
throughput alone is insufficient. Report code and prose separately and account
for prefill cost at long prompt depths.

## Strix Vulkan v0.7.5 matrix

| Profile | Head | Draft width |
| --- | --- | ---: |
| `qwen3.8-flash-next-ud-q4-k-xl-strix075-baseline` | None | — |
| `qwen3.8-flash-next-ud-q4-k-xl-strix075-mtp-n2` | Q4_K_M | 2 |
| `qwen3.8-flash-next-ud-q4-k-xl-strix075-mtp-advanced` | Q4_K_M | 3 |
| `qwen3.8-flash-next-ud-q4-k-xl-strix075-mtp-n4` | Q4_K_M | 4 |
| `qwen3.8-flash-next-ud-q4-k-xl-strix075-mtp-q8-n2` | Q8_0 | 2 |
| `qwen3.8-flash-next-ud-q4-k-xl-strix075-mtp-q8-n4` | Q8_0 | 4 |

All six use 32K context, batch/ubatch 2048/512, Q8 target K/V, one slot, mmap,
lazy tensor reads, disabled repacking/host buffers, and a 2,048-token medium
reasoning budget. MTP arms use probability floor 0.75 and Q8 draft K/V.

```bash
halo-ai profiles acquire qwen3.8-flash-next-ud-q4-k-xl-strix075-baseline
halo-ai profiles acquire qwen3.8-flash-next-ud-q4-k-xl-strix075-mtp-n2
halo-ai start qwen3.8-flash-next-ud-q4-k-xl-strix075-baseline --switch
halo-ai test qwen3.8-flash-next-ud-q4-k-xl-strix075-baseline
halo-ai start qwen3.8-flash-next-ud-q4-k-xl-strix075-mtp-n2 --switch
halo-ai test qwen3.8-flash-next-ud-q4-k-xl-strix075-mtp-n2
```

## Interpretation of the recorded screen

Target-only remains the reference for this lane. Q4 width 4 is a performance
option for long generation after shallower prompts: a 4K+1,024 request was
35.8% faster end-to-end, while a 31,743+1,024 request was 0.94% slower. Its
speculative prefill cost exceeded its decode benefit near the 32K ceiling.
Q8 width 4 was 4.5–4.8% slower than Q4 by two-pass wall-time median and used
about 0.8–0.9 GiB more GTT.

Both Q4 and Q8 matched only 12/13 target token streams despite equal 10/13
quality scores. They remain experimental performance profiles. These are small
fixed screens, not general quality or workload rankings. The
[result record](results/qwen3.8-flash-next-strix075-advanced-2026-09-10.json)
contains the detailed timings and outputs.

The smaller Q6 XL family uses DFlash2 and its own qualification controls. Its
sidecar is incompatible with Flash-Next's `qwen4exp` shared MTP head. Settings
from EngramHalo or custom ROCm forks also require separate runtime integration;
a launch recipe does not transfer their kernel changes into these pinned builds.
