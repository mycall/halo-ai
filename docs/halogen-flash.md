# Halogen Flash-Next: current decision and runbook

## Conclusion — 2026-10-05

**Keep v2 as the current serving checkpoint for its lower memory use. Our local
comparison found no meaningful generation-speed gain. Overall model quality
is still unproven by the tests we have run.**

- **Memory:** v2 held 8.7 GiB less engine memory at the matched 32k setup.
- **Speed:** serial generation was effectively unchanged; MTP was about 2% slower
  in this small sample, with overlapping run ranges.
- **Quality:** v2 answered one more of our 13 fixed questions correctly. That is
  a regression-screen result, not evidence of a general coding/reasoning gain.
- **Long context:** both checkpoints missed one of three codes near 262k tokens.
  A configured 262k context does not establish reliable retrieval at that length.
- **Last validated service:** v2, 262k context, vision, NPU embeddings and
  reranking. All restored-service checks passed.

The current choice is a practical memory decision. Keep W4B available as a
reference until broader quality tests are complete.

[Measured comparison](#measured-comparison) · [Better tests](#better-tests-to-run-next) ·
[Run and select a profile](#run-and-select-a-profile) ·
[Evidence and reproduction](#evidence-and-reproduction) ·
[Historical test/setup notes](halogen-flash-history.md)

## Measured comparison

Same pinned Halogen 0.16.2 engine and host; W4B with the quality overlay versus
v2. NPU and vision were disabled during comparison. Requests used greedy
decoding, seed 1, thinking off, cache off, and prompt lookup off.

| Measurement | W4B + quality overlay | V2 | Interpretation |
| --- | ---: | ---: | --- |
| Serial generation, median of 3 | 35.6 tok/s | 35.6 tok/s | No measured gain |
| MTP generation, median of 3 | 58.6 tok/s | 57.5 tok/s | No clear advantage |
| Locked weights | 68.0 GiB | 62.1 GiB | V2 saves 5.9 GiB |
| Total engine memory at 32k | 79.5 GiB | 70.8 GiB | V2 saves 8.7 GiB |
| Local 13-question screen | 10/13 | 11/13 | Too small for a quality ranking |
| Retrieval, 31,724 prompt tokens | 3/3 codes | 3/3 codes | Both pass this fixture |
| Retrieval, 261,107 prompt tokens | 2/3 codes | 2/3 codes | Known miss persists |

V2 fixed the modular-arithmetic question; both code questions still failed.
Within each checkpoint, serial and MTP returned identical text on all 13 cases
and the 512-token generation probe. That supports MTP integration correctness
on these inputs; it does not prove fidelity to the original model.

**Limits of this comparison:** one generation prompt, three timed runs per
mode, one retrieval document per length, and W4B always tested first. The first
timed run was slower for both checkpoints. Generated code differed between
checkpoints despite the same 512-token budget. Memory figures are reported by
the engine at 32k, excluding the demand-paged lookup table and other processes;
they are not total system RAM use for the 262k + NPU serving profile.

## Better tests to run next

**The existing suite is useful for quick regression checks, but insufficient
for choosing the better model.** Prioritize a larger coding evaluation and a
multi-task long-context evaluation before making a quality recommendation.

### What the Reddit discussion actually suggests

In the visible comments of the [linked 0.16.0 announcement](https://www.reddit.com/r/StrixHalo/comments/1wvfrj0/halogenflashserver_0160_strix_halo_npu_now_serves/),
I found no named evaluation-suite recommendation. Related Halogen discussions
provide more concrete leads:

- A [comment on inference fidelity](https://www.reddit.com/r/StrixHalo/comments/1w7gu6f/comment/p7xgnd9/)
  asks for **perplexity and KL divergence against BF16**. A
  [follow-up](https://www.reddit.com/r/StrixHalo/comments/1w7gu6f/comment/p84izmg/)
  asks for **top-1 token agreement on the same corpus**. These measure deviation
  from a reference model; they are not task-success benchmarks.
- A [related 12-quantization benchmark post](https://www.reddit.com/r/StrixHalo/comments/1wl3weh/i_benchmarked_12_quantizations_of_qwen38flashnext/)
  uses **HumanEval+**, 164 coding problems, and reports base and extended-test
  pass@1. This is a benchmark used in a related post, not a suggestion in the
  original announcement's comments. Its historical engine/settings results
  should not be carried over to our 0.16.2 installation.

### Proposed evaluation order — not yet run locally

| Priority | Evaluation | What it would add |
| --- | --- | --- |
| 1 | [EvalPlus: HumanEval+ and MBPP+](https://github.com/evalplus/evalplus) | Executable tests of generated code across many problems |
| 2 | [RULER](https://github.com/NVIDIA/RULER), then [LongBench v2](https://github.com/THUDM/LongBench) | Multiple long-context tasks and documents, beyond our repeated-record fixture |
| 3 | BF16-reference perplexity, KL divergence, top-1 agreement | Direct quantization/runtime fidelity evidence, if compatible reference outputs and scoring access are available |
| Separate NPU track | [MTEB](https://github.com/embeddings-benchmark/mteb) retrieval/reranking tasks | Dataset-level relevance quality beyond vector shape and one obvious ranking |

EvalPlus is the concrete community benchmark lead. RULER, LongBench v2, and
MTEB are our proposed follow-ups based on the gaps in the local tests, not
claims about what the linked commenters recommended.

For a fair comparison, pin datasets and evaluator versions, use identical
prompts and token budgets, verify reasoning settings, and report truncations
separately. Test thinking off and the actual serving reasoning policy as
separate conditions. For coding, score generated code in an isolated runner.
Report task success and time per completed task alongside tokens/s. Fidelity
against BF16 needs a suitable reference; agreement between two quantized runs
alone is not that measurement.

## Run and select a profile

**Use the explicit v2 alias.** `qwen3.8-fn` still selects W4B; switching the
running server did not change aliases.

| Alias | Checkpoint | Context | Vision | NPU services |
| --- | --- | ---: | --- | --- |
| `qwen3.8-halogen-v2-npu` | V2 | 262,144 | Yes | Embeddings + reranking |
| `qwen3.8-halogen-v2` | V2 | 262,144 | Yes | Off |
| `qwen3.8-halogen-npu` | W4B + quality overlay | 262,144 | Yes | Embeddings + reranking |
| `qwen3.8-fn` / `qwen3.8-halogen` | W4B + quality overlay | 262,144 | Yes | Off |

```bash
# Preview the full checkpoint, vision, and NPU dependency selection.
halo-ai profiles acquire qwen3.8-halogen-v2-npu --dry-run
# Download and verify missing artifacts, then install the pinned runtime.
halo-ai profiles acquire qwen3.8-halogen-v2-npu
halo-ai start qwen3.8-halogen-v2-npu --switch
halo-ai test qwen3.8-halogen-v2-npu
```

Acquisition includes selected embedding/reranker weights and their shared device
programs. The plan lists these under `auxiliary`; v2 selects the external n-gram
table, while W4B selects the quality overlay. Host NPU setup remains a separate
prerequisite described below.

- OpenAI-compatible base URL: `http://127.0.0.1:8731/v1`
- V2 model ID: `halogen-qwen3.8-flash-next-v2`
- Embedding model: `qwen3-embedding-0.6b`, route `/v1/embeddings`
- Reranker: `qwen3-reranker-0.6b`, route `/v1/rerank`

Serving uses one slot, MTP plus prompt lookup, session cache mode 2, and
16,384-token prefill chunks. Defaults are medium reasoning, 8,192 completion
tokens, temperature 1.0, top-p 0.95, and top-k 20; explicit request fields win.
Reasoning consumes the completion budget. Diagnostic profiles disable cache
and prompt lookup. Runtime startup is explicit, not automatic at boot.

## NPU and host constraints

The last validated host has IOMMU enabled, the `amdxdna` driver, XRT libraries,
an unlimited operator memlock limit, and a 118 GiB GTT aperture. The upstream
fabric-clock helper holds FCLK at 2000 MHz. Its oneshot service may show
`inactive (dead)` after success; check `halogen-fabric-clock status`.

**NPU work shares resources with GPU generation.** Five alternating v2 probe
pairs measured median GPU decode of 50.48 tok/s with the NPU idle and 39.149
tok/s under continuous embedding/reranking traffic: a 22.4% reduction. All ten
512-token outputs were identical, and concurrent NPU vectors/scores matched
their idle references. This is one fixed prompt with the serving cache and
prompt lookup enabled, not a general slowdown estimate. The first idle prompt
was cold; subsequent probes reused its 38-token prefix. Idle decode ranged
46.3–58.0 tok/s, versus 38.9–39.6 under load. The earlier W4B test measured
about 26% slowdown under its own conditions.

Synthetic short-input batches reached about 9,232 embedding tokens/s and
18.74 rerank pairs/s; longer inputs crossed a much slower execution bucket.
These are throughput observations, not retrieval-quality scores.

Embeddings and reranking passed API and concurrent correctness checks. Gaming,
suspend/resume, and extended stability have not been qualified. IOMMU was
changed for NPU compatibility; no gaming benefit was measured.
[Host setup, memlock repair, and detailed NPU results](halogen-flash-history.md#optional-npu-embeddings-and-reranking)
are retained in the history.

### V2 ingestion follow-up (2026-10-05)

The pinned 0.16.2 v2/vision/NPU profile passed these bounded bursts without
client retries. The GPU model stayed loaded but was idle during ingestion.
The 188-file tests used the first 188 SciFact corpus documents in sorted ID
order, split into at most 120-word chunks (444 inputs total).

| Workload | Successful requests | Total wall time | Request latency p95 |
| --- | ---: | ---: | ---: |
| Issue #127 pattern: 40 concurrent requests, 10 inputs each | 40/40 | 33.3 s | 31.6 s |
| 188 files, 12 concurrent requests | 188/188 | 75.0 s | 5.2 s |
| 188 files, 40 concurrent requests | 188/188 | 74.7 s | 16.2 s |

Every response had complete, finite, normalized vectors. The issue-pattern
vectors also matched an idle reference. The service remained healthy with no
NPU work queued after the test. Increasing concurrency from 12 to 40 bought
essentially no throughput for this workload and increased request latency.
This is an HTTP ingestion simulation, not an Open WebUI integration test or
proof that arbitrary bursts cannot hit the queue timeout.

Evidence: [paired v2 contention](results/halogen-0162-v2-npu-evaluation-2026-10-05/contention.json),
[ingestion requests and timings](results/halogen-0162-v2-npu-evaluation-2026-10-05/ingestion.json),
[runtime and hardware](results/halogen-0162-v2-npu-evaluation-2026-10-05/environment.json).

The follow-up runner uses the public
[BEIR SciFact archive](https://github.com/beir-cellar/beir/wiki/Datasets-available)
and verifies SHA-256 `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`.
The archive also matches BEIR's published MD5. Keep the dataset and embedding
cache outside the checkout. The quality mode requires NumPy; this host already
had it installed. Run the stages sequentially with no other inference clients:

```bash
mkdir -p /var/cache/halo-ai/benchmarks/scifact
curl --fail --location \
  https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip \
  --output /var/cache/halo-ai/benchmarks/scifact/scifact.zip
halo-ai start qwen3.8-halogen-v2-npu --switch
python3 tools/halogen_npu_validate.py --rounds 5 --output /tmp/v2-contention.json
python3 tools/halogen_npu_evaluate.py burst --output /tmp/v2-ingestion.json
python3 tools/halogen_npu_evaluate.py quality --output /tmp/v2-scifact.json
```

Quality mode embeds the complete corpus and evaluates every test query. Its
resumable corpus cache is keyed by dataset hash, document formatting, embedding
model/dimensions, runtime version, and the pinned NPU artifact manifest. Query
embeddings and reranking are recomputed. Raw rankings and per-query metrics are
saved with the result; no corpus text or embedding cache is added to this repo.

### NPU retrieval quality: SciFact (2026-10-05)

The full BEIR SciFact test split covers **300 judged queries and 5,183
documents**. We embedded complete titles and abstracts without truncation,
used the task instruction "Given a scientific claim, retrieve documents that
support or refute the claim", and ranked by cosine similarity of normalized
1024-dimensional NPU vectors. The reranker then reordered each query's top ten
embedding candidates using the same task instruction. GPU v2 stayed loaded
but idle. Corpus embedding took 373.9 seconds; mean top-ten reranking request
latency was 0.947 seconds. No embedding or reranking request failed.

| Method | nDCG@10 | MRR@10 | Recall@10 | Recall@100 |
| --- | ---: | ---: | ---: | ---: |
| Local BM25 baseline | 0.6617 | 0.6276 | 79.09% | 88.59% |
| NPU embeddings | 0.7001 | 0.6596 | 84.27% | 95.33% |
| NPU embeddings + top-ten reranking | 0.7504 | 0.7251 | 84.27% | 95.33% |

nDCG measures relevance ordering; MRR measures the reciprocal position of the
first relevant result. Recall is the per-query fraction of judged relevant
documents retrieved, averaged across queries. Higher is better for each.
Reranking preserves recall@10 because it only reorders those same candidates.
It improved nDCG@10 on 66 queries, worsened 28, and left 206 unchanged. The
mean paired gain was **0.0503**, with a **0.0292–0.0717** percentile bootstrap
95% interval (5,000 resamples of query differences, seed 1).

This supports the NPU reranker on this dataset. It is not full MTEB
qualification, a BF16 fidelity test, or a quality result for the Flash-Next
generation model. The BM25 baseline uses k1=1.2, b=0.75, lowercase Unicode
word tokens, and no stemming or stopword removal; it is a local comparator,
not a reproduction of a published BM25 score. Longer candidate lists, other
domains, and application-specific documents remain unmeasured.

[Complete rankings, per-query scores, and methods](results/halogen-0162-v2-npu-evaluation-2026-10-05/scifact.json)
are retained for reproduction and failure analysis. The evaluator's metric
tests and the complete container suite passed: 175 tests plus shell/CLI/
installer smoke checks.

After the workload, vision returned `red`, all NPU correctness/concurrency
checks passed again, and the embedding reference hash matched the pre-workload
value. The kernel journal contained no matching GPU/XDNA/IOMMU fault or OOM
reports during the evaluation, and the container reported no OOM kill. The
runtime was stopped afterward to restore its initial stopped state.
[Evaluation summary](results/halogen-0162-v2-npu-evaluation-2026-10-05/summary.json),
[post-load NPU checks](results/halogen-0162-v2-npu-evaluation-2026-10-05/post-load-npu.json),
and [host checks](results/halogen-0162-v2-npu-evaluation-2026-10-05/post-load-host.json)
record the checks and their scope.

## Artifacts and runtime pin

Model directory: `/srv/halo-ai/models/peonist-ai/halogen-qwen3.8-flash-next/`.
Both v2 files are downloaded and fully SHA-256 verified:

| File | Size |
| --- | ---: |
| `qwen38-flash-next-v2.hgn` | 66,687,678,432 bytes |
| `qwen38-flash-next-ngram.hgn` | 51,200,246,144 bytes |

V2 uses the external n-gram table and reuses the verified tokenizer and vision
files. It does not mount the W4B quality overlay. Both native checkpoint
formats supply their own MTP head; the separate MTP HGN is for the GGUF path.

- Runtime: Halogen **0.16.2**, closed-source engine with public deployment code.
- Image digest: `sha256:0c61bf84ac22308a53f5d1ca6b86806702d7039e5ebc51cae4c66621b92fe04a`.
- V2 source revision: `3648cf1e6a3143e8946d52301570f26003c0bdb0`.
- W4B source revision: `e238ffde1a89531743505349dd16ca35a1e09b15`.
- [Pinned download links](halogen-flash-history.md#download-v2-separately) and
  [full verification](results/halogen-0162-v2-verification-2026-10-05.json).

## Evidence and reproduction

| Evidence | Record |
| --- | --- |
| Comparison, settings, timings, and outputs | [Comparison JSON](results/halogen-0162-checkpoint-comparison-2026-10-05/comparison.json) |
| Fixed correctness suite | [W4B](results/halogen-0162-checkpoint-comparison-2026-10-05/w4b-quality.json), [v2](results/halogen-0162-checkpoint-comparison-2026-10-05/v2-quality.json) |
| Retrieval prompts and answers | [W4B](results/halogen-0162-checkpoint-comparison-2026-10-05/w4b-retrieval.json), [v2](results/halogen-0162-checkpoint-comparison-2026-10-05/v2-retrieval.json) |
| Restored service checks | [Vision](results/halogen-0162-checkpoint-comparison-2026-10-05/restored-smoke.json), [NPU](results/halogen-0162-checkpoint-comparison-2026-10-05/restored-npu.json) |
| Earlier NPU load and throughput | [Paired GPU/NPU load](results/halogen-0162-npu-2026-10-04.json), [NPU throughput](results/halogen-0162-npu-throughput-2026-10-04.json) |
| Dated setup, previous comparisons, and troubleshooting | [History](halogen-flash-history.md), [upstream review](halogen-upstream-review.md) |

To repeat the **small diagnostic comparison** from this checkout (temporarily
switches models, then restores v2 with NPU services):

```bash
python3 tools/halogen_checkpoint_compare.py --output-dir /tmp/halogen-checkpoint-comparison
```

The broader evaluations proposed above have not been run by this command.
