# Halogen Flash-Next

[Reference manual](README.md) · [Profiles](profiles.md)

Halogen serves Flash-Next with MTP, vision, and optional NPU embeddings and
reranking. The recommended aliases select the v2 checkpoint on the pinned
runtime. V2 uses a separate demand-paged n-gram table; broader generation quality
and reliable near-limit retrieval require workload-specific evaluation.

## Run and select a profile

| Alias or profile | Checkpoint | Context | Vision | NPU services |
| --- | --- | ---: | --- | --- |
| `qwen3.8-fn` / `qwen3.8-halogen` / `qwen3.8-halogen-v2` | V2 | 262,144 | Yes | Off |
| `qwen3.8-halogen-npu` / `qwen3.8-halogen-v2-npu` | V2 | 262,144 | Yes | Embeddings + reranking |

Legacy W4B profiles remain in the catalog for historical use, but their checkpoint
and overlays are not dependencies of the recommended v2 profiles. The upstream
[model card](https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next)
identifies v2 as the default and W4B as legacy.

Complete [installation](installation.md), then acquire and test the GPU profile:

```bash
halo-ai profiles acquire qwen3.8-fn --dry-run
halo-ai profiles acquire qwen3.8-fn
halo-ai start qwen3.8-fn --switch
halo-ai test qwen3.8-fn
```

For NPU service, complete the host setup below and use `qwen3.8-halogen-v2-npu`
in the same sequence. Acquisition includes the selected embedding/reranker files
and shared device programs; its preview lists them under `auxiliary`. Starting a
profile does not download missing weights.

After updating the checkout, use `./reload.sh` to deploy it. The installer
migrates the known old Halogen image pin while preserving custom pins.

## API and serving defaults

| Setting | Value |
| --- | --- |
| OpenAI-compatible base URL | `http://127.0.0.1:8731/v1` |
| V2 model ID | `halogen-qwen3.8-flash-next-v2` |
| Embedding model and route | `qwen3-embedding-0.6b`, `/v1/embeddings` |
| Reranker and route | `qwen3-reranker-0.6b`, `/v1/rerank` |
| Parallel slots | 1 |
| Speculation | MTP plus prompt lookup |
| Session cache | RAM cache mode 2 |
| Prefill chunk | 16,384 tokens |
| Reasoning / completion budget | `medium` / 8,192 tokens |
| Sampling | Temperature 1.0, top-p 0.95, top-k 20 |

Explicit request fields override serving defaults. Reasoning consumes the
completion budget. Diagnostic profiles disable cache and prompt lookup to
isolate MTP behavior. Runtime startup is explicit, not automatic at boot.

The prefill chunk controls how much prompt input is processed at a time;
longer prompts span multiple chunks within the 262,144-token context window.
The 16,384-token setting limits temporary working memory and leaves desktop
headroom. It is a configured default, not a measured optimum or an input-length
limit.

Embedding vectors have 1024 dimensions and are normalized. Format retrieval
queries as `Instruct: ...\nQuery:...`. The rerank 4096-token limit includes the
query, document, and template. Oversized NPU inputs are rejected.

The serving screen covers schema/tool use, nullable arguments, streamed tool
calls, and late system/developer messages. Late instructions are accepted as
user text, not as privileged system messages. Startup requires a successful
engine capability probe; NPU profiles additionally require healthy NPU status
and every selected model in both health information and `/v1/models`.

## NPU and host constraints

NPU service requires:

- firmware IOMMU enabled
- the `npu` boot profile
- `amdxdna` driver
- XRT libraries
- unlimited operator memlock limit
- fabric-clock helper.

The reference configuration retains 118 GiB GTT.

After host installation, preview and apply preparation:

```bash
bash /opt/halo-ai/current/tools/halogen_npu_host_setup.sh --dry-run
pkexec /usr/bin/bash /opt/halo-ai/current/tools/halogen_npu_host_setup.sh
```

The helper installs CachyOS `xrt` and `xrt-plugin-amdxdna`, configures operator
memlock, installs checksum-pinned fabric-clock files, enables their service,
and stages the NPU boot profile with snapshots and verified boot backups. It
preserves GTT/TTM and does not reboot. Reboot manually with firmware IOMMU enabled,
then check and acquire the profile:

```bash
halo-ai host-profile status
halogen-fabric-clock status
ulimit -H -l
halo-ai profiles acquire qwen3.8-halogen-v2-npu
python3 /opt/halo-ai/current/tools/halogen_npu_prepare.py
halo-ai start qwen3.8-halogen-v2-npu --switch
halo-ai test qwen3.8-halogen-v2-npu
```

The fabric-clock helper holds FCLK at 2000 MHz. Its oneshot service can report
`inactive (dead)` after success; check the helper's status. It remains enabled
across boots and resume, including after switching to GPU-only mode. To undo
that host change, disable its service and run `halogen-fabric-clock release`
as root.

Existing login sessions can retain their old hard memlock limit. Renew the login
if `ulimit -H -l` is still limited; container `--ulimit` cannot raise the host
process's hard limit. The standalone repair command is:

```bash
pkexec /usr/bin/python3 /opt/halo-ai/current/tools/halogen_memlock_setup.py --user "$USER"
```

XRT mounts resolve real library files and expose them read-only. Set
`HALOGEN_XRT_LIB_DIR` if the host libraries are outside `/usr/lib`. Model mounts
also remain read-only; serving does not download artifacts or mount writable
host sysfs. Gaming, suspend/resume, and extended stability are not qualified by
the bounded inference checks.

## Artifacts and runtime pin

The authoritative runtime pin is in the [configuration template](../config/halo-ai.env.example)
and [engine policy](../lib/halo_ai/engine_halogen.py). These record the release
and immutable image digest used by this checkout. Use `halo-ai status` to
inspect the configured image and running service.

Model directory: `/srv/halo-ai/models/peonist-ai/halogen-qwen3.8-flash-next/`.

| V2 artifact | Size |
| --- | ---: |
| `qwen38-flash-next-v2.hgn` | 66,687,678,432 bytes |
| `qwen38-flash-next-ngram.hgn` | 51,200,246,144 bytes |

V2 uses the external n-gram table and shared tokenizer/vision artifacts. Keep
the n-gram file alongside the checkpoint. The native v2 checkpoint includes its
own MTP head; the separate MTP HGN serves the GGUF path. Sizes and hashes are recorded in the
[catalog](../config/models.d/halogen.json) and [expected-file manifest](halo-ai.md#expected-file-manifest).
Use profile acquisition to select the required files instead of downloading
all catalog entries.

The [NPU manifest](../lib/halo_ai/halogen_npu_models.json) tracks runtime and
artifact releases separately. NPU files live under
`/srv/halo-ai/models/halogen-npu/<artifact_release>`, using the manifest's
`artifact_release` value. Runtime upgrades can require new device programs even
when model weights are unchanged; acquire the profile before starting the new
runtime. The engine is closed source; public deployment code and local validation
establish the integration boundary.

## Qualification and limitations

The current qualification uses the pinned runtime and native v2 checkpoint on
the reference 128 GiB host, with 118 GiB GTT, 2 GiB firmware VRAM, the performance
power profile, and FCLK held at 2000 MHz. The evidence records the exact release,
image digest, settings, and methods. Rerun the relevant checks after changing the
runtime, artifacts, host, or profile.

Reported SoC power stayed near 45 W during GPU load. This is observed power,
not a readback of the firmware TDP setting. Treat these speeds as specific to
that power envelope; comparisons with earlier runs at unverified power limits
cannot isolate runtime performance changes. The concurrent GPU/NPU test keeps
the same host settings, but does not separate power sharing from memory contention.

| Area | Measured result | Qualification boundary |
| --- | --- | --- |
| Serving | All 14 serving checks passed | Passing fixtures does not qualify arbitrary client workflows |
| Diagnostic MTP | Serial and MTP matched on all 13 cases and the 512-token probe; both scored 11/13, with active draft acceptance | The alternating-code trace and slice-semantics questions still failed; agreement does not establish general coding quality |
| Diagnostic decode | Three paired 512-token runs measured median 34.9 tokens/s serial and 65.7 tokens/s MTP, with identical output | Cache, prompt lookup, vision, NPU, and thinking were off; this is one fixed prompt at 32K configured context |
| Serving performance | Median decode was 50–52 tokens/s for short prompts and 44–48 tokens/s for 31.7K prompts; long-prompt wall time fell from about 46 seconds to 11 seconds on cache repeats | Three rounds per prompt length, sampling mode, and cache pass; 512 output tokens, thinking off, vision/NPU loaded but idle |
| Engine memory | At 32K: 62.1 GiB locked weights + 0.9 GiB KV + 8.1 GiB working memory = 71.1 GiB | Excludes the 47.7 GiB demand-paged n-gram table and other host processes; this is not total 262K/NPU memory |
| Long-context retrieval | 31,724-token control: 3/3 codes; 261,107-token near-limit prompt: 2/3 | One request at each depth; the near-limit miss persists, so configured context is not a retrieval guarantee |
| Concurrent GPU/NPU | All 8 NPU checks and 10 GPU probes passed; median decode fell from 53.0 to 41.6 tokens/s under continuous NPU load, about 22% | Five alternating idle/busy pairs on one prompt; contention depends on prompt, cache, and NPU workload |
| NPU ingestion | All 416 requests passed. Embedding 188 chunked abstracts took 41.5 seconds at both 12 and 40 workers; median request latency rose from 2.66 to 8.75 seconds | More concurrency gave no useful throughput gain in this workload; application integration needs separate testing |
| Retrieval quality | On 300 SciFact queries over 5,183 freshly embedded documents, nDCG@10 was 0.700 for embeddings and 0.750 after top-ten reranking; BM25 scored 0.662 | One dataset, not full MTEB or generation-model quality; reranking the same ten candidates cannot improve recall@10 |

Performance screens measure a small set of prompts, output lengths, and cache
conditions. Their results do not establish general throughput or quality at the
context limit. Exact timings, scores, and comparison methods remain in the
evidence records below.

## Validation commands and evidence

Run the HTTP screens against an otherwise idle active instance:

```bash
python3 tools/halogen_serving_validate.py --output /tmp/halogen-serving.json
python3 tools/halogen_performance_validate.py --rounds 2 --output /tmp/halogen-performance.json
# Requires an active NPU profile:
python3 tools/halogen_npu_validate.py --rounds 5 --output /tmp/halogen-npu.json
```

For a focused MTP check, switch to the v2 diagnostic profile. This disables
vision, NPU services, cache, and prompt lookup while comparing serial and MTP
on the fixed quality suite:

```bash
halo-ai start qwen3.8-flash-next-halogen-v2-32k-mtp --switch
python3 tools/halogen_validate.py --output /tmp/halogen-mtp.json
halo-ai start qwen3.8-halogen-v2-npu --switch
```

The complete [requalification runner](results/halogen-requalification-2026-10-10/run.py)
also measures near-limit retrieval, repeated decode, GPU/NPU contention, ingestion,
and SciFact. It requires an idle container host, the prepared NPU host and dataset,
and acquired v2 artifacts. It uses the checkout's `.halo-ai-workspace.env` and
stops the runtime afterward. Choose a fresh output directory:

```bash
python3 docs/results/halogen-requalification-2026-10-10/run.py \
  --output-dir /tmp/halogen-requalification
python3 docs/results/halogen-requalification-2026-10-10/summarize.py \
  --input-dir /tmp/halogen-requalification
```

For dataset acquisition and NPU evaluation, see [Benchmarks](benchmarking.md#npu-retrieval-evaluation).
These screens do not run broader coding or long-context quality benchmarks.

| Evidence | Record |
| --- | --- |
| Runtime, host, method, and completion status | [Run record](results/halogen-requalification-2026-10-10/run-state.json) |
| Consolidated measurements | [Summary](results/halogen-requalification-2026-10-10/summary.json) |
| V2 MTP, decode, memory, and retrieval | [Diagnostics](results/halogen-requalification-2026-10-10/v2-diagnostics/summary.json), [memory log](results/halogen-requalification-2026-10-10/v2-diagnostics/32k-container.log) |
| Serving and cache performance | [API checks](results/halogen-requalification-2026-10-10/serving.json), [request timings](results/halogen-requalification-2026-10-10/performance.json) |
| V2 GPU/NPU contention | [Paired measurements](results/halogen-requalification-2026-10-10/npu-contention.json) |
| NPU ingestion bursts | [Request timings](results/halogen-requalification-2026-10-10/ingestion.json) |
| SciFact methods, rankings, and per-query scores | [Evaluation](results/halogen-requalification-2026-10-10/scifact.json) |
