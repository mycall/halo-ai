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
