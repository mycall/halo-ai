# Halogen upstream review, 2026-10-04

Reviewed the public deployment changes from `v0.12.2` through `v0.16.2`,
pinned to [7f31bbd4021f217a1be9776bdb7304bcf8eca62d](https://github.com/peonist-ai/halogen-flash-server/commit/7f31bbd4021f217a1be9776bdb7304bcf8eca62d).
The engine/API implementation is private; mechanisms described in its changelog
are upstream claims, with our bounded validation recorded separately.

The strongest upgrade reasons for this project are agent-serving fixes:
0.13.4 handles end-of-turn markers and tool calls inside reasoning;
0.13.5/0.13.6 combine split assistant text/tool calls when replaying history;
0.14.0 emits the initial streaming role; and 0.14.2 handles images in Responses
tool results. Our new serving screen exercises the latter three behaviors.
0.13.1 also fixes capability-probe fallback; Halo now rejects a health response
unless that probe succeeded, rather than trusting potentially fallback limits.

0.14.0 rewires the MTP head and proposes two tokens per round. 0.16.0 fixes its
proposal counters, so acceptance must use the corrected counters. Serial/MTP
equality is a within-build property: later prefill kernels change rounding and
sampled seeds are not an across-version equivalence test. Our 0.16.2 screen
keeps greedy decoding, no cache and no prompt lookup for this comparison.

0.15 changes the default to v2 with a separate n-gram table. Explicit W4B
selection remains supported, so we upgraded the runtime using our verified old
weights. Separate v2 profiles and pinned artifact records are ready, but their
local evaluation awaits the two downloads. The optional ht43 checkpoint trades
precision/memory and speed differently; it is not selected for this upgrade.

0.16.1 fixes disk-cache restoration and queue handling; 0.16.2 changes cache
eviction. Our profile uses RAM cache mode 2, with no persistent cache directory.
The API smokes do not qualify multi-conversation cache pressure or cancellation
latency. We retain one slot and the existing 262K pool / 16K prefill allocation.

0.16 introduces NPU side models, and 0.16.2 adds moderation and small generation.
The first integrated NPU profile selects embeddings plus reranking. Its files
and shared device program are pinned to the image's manifest. Host XRT library
loading passed in the image. After reboot, both NPU routes and bounded
concurrent GPU/NPU tests passed; the rootless login also needed a raised
memlock limit, now checked and included in host preparation. The rootless setup requires the upstream fabric-clock service; silently
bypassing the clock guard is not part of the integration. This also means GPU
performance must be remeasured after the host-profile change.

The previous near-262K retrieval finding remains evidence about the old test
conditions, not a general explanation of model quality. Updated measurements
and exact limitations belong in [the current runbook](halogen-flash.md).

Sources: [release changes](https://github.com/peonist-ai/halogen-flash-server/compare/v0.12.2...v0.16.2),
[pinned changelog](https://github.com/peonist-ai/halogen-flash-server/blob/7f31bbd4021f217a1be9776bdb7304bcf8eca62d/CHANGELOG.md),
[pinned NPU guide](https://github.com/peonist-ai/halogen-flash-server/blob/7f31bbd4021f217a1be9776bdb7304bcf8eca62d/docs/NPU.md).

---

# Halogen upstream review, 2026-09-20

Reviewed peonist-ai's recent public Reddit post/comment feeds and followed
their technical discussions into the release documentation and dated GitHub
issue replies. Reddit's indexed pages have different crawl ages; this is a
review of accessible recent material, not an exhaustive account export.

## Accuracy leads

**Sparse-attention budget is the strongest new lead.** In the
[September 15 maintainer reply to issue #57](https://github.com/peonist-ai/halogen-flash-server/issues/57#issuecomment-5674392216),
raising `HALOGEN_INDEXER_BUDGET` from 2048 to 4096 recovered two known
retrieval misses but introduced a different early-stop failure: 95/96 overall
versus 94/96. At 8192 the score was 96/96, with larger speed and perplexity
costs. This changes the attention computation, not just cache capacity.

Our installed 0.12.2 reports `indexer_budget: 2048`. The existing UD-Q4_K_XL
GGUF likewise declares `qwen4exp.attention.indexer.top_k: 2048`. Both pairs
missed the same requested record in the prior serial near-262K screen.
This supports testing attention selection as a common factor; it does not
establish the cause of either failure.

**An older coding-quality claim was withdrawn.** The
[September 12 follow-up to issue #16](https://github.com/peonist-ai/halogen-flash-server/issues/16#issuecomment-5647792921)
corrects the reported repeated feature omissions after matching sampling.
Those particular omissions disappeared. Output budgets were still unmatched,
and behavioral defects remained, so this was not a clean general superiority
result for either runtime. Our serving defaults already use temperature 1.0,
top-p 0.95, top-k 20, medium effort, and 8192 total output tokens. Our greedy,
thinking-off screens deliberately measure a different operating point.

**The cache-mode bug is already fixed in our version.** The
[September 17 resolution of issue #65](https://github.com/peonist-ai/halogen-flash-server/issues/65#issuecomment-5707695512)
identifies a mode-2 cold-prefill arithmetic difference fixed in 0.11.3.
Continuing from a reused prefix can still change numerical results; cache
mode 1 is the exact-repeat option. Our 0.12.2 failure also occurs with cache
mode 0 and zero cached tokens, so that bug does not explain the serial miss.

**Use the same GGUF to separate runtime from weights.** Peonist-ai's
[BYO GGUF announcement](https://www.reddit.com/r/StrixHalo/comments/1wf9joo/byo_gguf_to_halogen_070/)
and the [maintainer's issue #16 follow-up](https://github.com/peonist-ai/halogen-flash-server/issues/16#issuecomment-5653976303)
make this a useful next control. Our existing comparison changes both runtime
and model representation. Native HGN versus UD GGUF inside Halogen would
hold the runtime fixed; the same UD file on both engines would hold weights
fixed. The catalog currently integrates Halogen's native HGN path only.

## Performance and release advice

The author's [recent comments](https://www.reddit.com/user/peonist-ai/comments/?sort=new)
mention disabling AMD IOMMU for prefill performance. This host already boots
with `amd_iommu=off`. The quality overlay and tuned matrix-multiplication
plan are also already active. Our one-slot, 262K-pool, 16K-prefill setup
reserves more host headroom than the upstream multi-slot defaults.

The [reviewed changelog](https://github.com/peonist-ai/halogen-flash-server/blob/d4e357a42acd24bfb361f2896f1b4989885cc36f/CHANGELOG.md)
adds 0.12.3 watchdog, repack, and logging fixes. It explicitly describes no
weight change or change to engine numerics; upgrading is not evidence of a
retrieval fix. The controlled attention experiment retains our pinned 0.12.2.

No BIOS, kernel, sampling-default, or image-pin changes were made for this
review. Attention-budget trials are temporary diagnostic configurations.

## Local attention-budget experiment

We tested the documented lever instead of assuming it fixed our failure.
Each budget used a fresh pinned 0.12.2 process, the same quality-overlay HGN,
and the same 261,107-token request. MTP, prompt lookup, caching, and thinking
were off; temperature was 0 and seed was 1. Context, prefill chunk size,
vision, and slot count stayed unchanged. The baseline is the earlier serial
run from this session.

| Attention budget | First code returned | Correct codes | Request time |
| ---: | --- | ---: | ---: |
| 2048 | `delta-92415` (record `000794`) | 2/3 | 279.74 s |
| 4096 | `delta-17739` (record `003790`) | 2/3 | 322.74 s |
| 8192 | `delta-17739` (record `003790`) | 2/3 | 432.07 s |

The requested first record was `000791`, with code `delta-68658`.
Every budget returned the correct middle and final codes, processed the
whole prompt, drafted zero tokens, used zero cached tokens, and stopped
normally. The larger budgets change the result but do not recover the fact.
This does not rule out sparse attention as a contributing cause; it rules
out claiming that these documented increases fix this particular screen.

All three budgets passed the 31,724-token three-code control and scored
10/13 on the fixed suite, with the same code-trace, code-slice, and CRT
failures. These are single-run timings, not a repeated performance study.

Keep the shipped budget at 2048. An increased budget remains an experiment
for other workloads, with a measurable cost and no local accuracy gain here.
The more informative next control is the existing UD GGUF inside Halogen,
holding runtime and request policy fixed while changing model representation.

[Full attention-budget results](results/halogen-attention-budget-2026-09-20.json)
record the effective health settings, requests, output and timing counters.
