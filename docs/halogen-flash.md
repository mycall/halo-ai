# Halogen Flash-Next

[Reference manual](README.md) · [Profiles](profiles.md)

Halogen serves Flash-Next with MTP, vision, and optional NPU embeddings and
reranking. The recommended aliases select the v2 checkpoint on pinned runtime
**0.17.3**. V2 reduces resident memory relative to W4B; broader generation quality
and reliable near-limit retrieval remain unqualified. Explicit W4B profiles are
retained for reference comparisons and are deprecated upstream.

## Run and select a profile

| Alias or profile | Checkpoint | Context | Vision | NPU services |
| --- | --- | ---: | --- | --- |
| `qwen3.8-fn` / `qwen3.8-halogen` / `qwen3.8-halogen-v2` | V2 | 262,144 | Yes | Off |
| `qwen3.8-halogen-npu` / `qwen3.8-halogen-v2-npu` | V2 | 262,144 | Yes | Embeddings + reranking |
| `qwen3.8-flash-next-halogen-262k-vision` | W4B + quality overlay | 262,144 | Yes | Off |
| `qwen3.8-flash-next-halogen-262k-vision-npu` | W4B + quality overlay | 262,144 | Yes | Embeddings + reranking |

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

Embedding vectors have 1024 dimensions and are normalized. Format retrieval
queries as `Instruct: ...\nQuery:...`. The rerank 4096-token limit includes the
query, document, and template. Oversized NPU inputs are rejected.

The serving screen covers schema/tool use, nullable arguments, streamed tool
calls, and late system/developer messages. Late instructions are accepted as
user text, not as privileged system messages. Startup requires a successful
engine capability probe; NPU profiles additionally require healthy NPU status
and every selected model in both health information and `/v1/models`.

## NPU and host constraints

NPU service requires firmware IOMMU enabled, the `npu` boot profile, the
`amdxdna` driver, XRT libraries, an unlimited operator memlock limit, and the
fabric-clock helper. The reference configuration retains 118 GiB GTT.

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
and [engine policy](../lib/halo_ai/engine_halogen.py):

```text
Halogen 0.17.3
ghcr.io/peonist-ai/halogen-flash-server@sha256:3bca0132db3c859c997d52d148e6ea4b7b497b8a695c5ccab97135193fde592a
```

Model directory: `/srv/halo-ai/models/peonist-ai/halogen-qwen3.8-flash-next/`.

| V2 artifact | Size |
| --- | ---: |
| `qwen38-flash-next-v2.hgn` | 66,687,678,432 bytes |
| `qwen38-flash-next-ngram.hgn` | 51,200,246,144 bytes |

V2 uses the external n-gram table and shared tokenizer/vision artifacts, without
the W4B overlay. Both native checkpoint formats include their own MTP head;
the separate MTP HGN serves the GGUF path. Sizes and hashes are recorded in the
[catalog](../config/models.d/halogen.json) and [expected-file manifest](halo-ai.md#expected-file-manifest).
Use profile acquisition to select the required files instead of downloading
all catalog entries.

The NPU manifest distinguishes runtime 0.17.3 from artifact release 0.16.2.
Artifact sizes and hashes match the runtime manifest, so files remain under
`/srv/halo-ai/models/halogen-npu/0.16.2`. That directory name does not select an
older server image. The engine is closed source; public deployment code and
local validation establish the integration boundary.

## Qualification and limitations

| Area | Established result | Limit |
| --- | --- | --- |
| Serving on 0.17.3 | 14/14 serving checks and 8/8 NPU checks passed | Bounded API fixtures, not arbitrary client workflows |
| Diagnostic MTP on 0.17.3 | Serial/MTP text matched on 13 fixed cases and a 512-token probe; both scored 11/13 | Two code-question failures remain; no general coding-quality claim |
| V2 memory on 0.16.2 | 70.8 GiB engine memory at 32K versus W4B's 79.5 GiB | Excludes demand-paged table and other processes; not total 262K/NPU memory |
| Checkpoint speed on 0.16.2 | Serial medians both 35.6 tok/s; MTP 57.5 v2 versus 58.6 W4B | One prompt, three runs, overlapping ranges |
| Retrieval on 0.16.2 | Both checkpoints recovered 3/3 codes at 31,724 tokens and 2/3 at 261,107 | Near-limit miss unresolved; not rerun on 0.17.3 |
| Concurrent GPU/NPU on 0.16.2 v2 | Five paired probes showed 22.4% median GPU decode reduction under continuous NPU load | One prompt with cache and prompt lookup; not a universal slowdown |
| NPU ingestion on 0.16.2 | Bounded 188-file bursts passed at concurrency 12 and 40 | Higher concurrency increased latency without useful throughput gain; not an Open WebUI integration test |
| SciFact on 0.16.2 | Embedding nDCG@10 0.7001; top-ten reranking 0.7504 across 300 queries / 5,183 documents | One retrieval dataset, not full MTEB or generation-model quality |

The 0.17.3 runtime comparison measured 8.7–21.2% higher median decode throughput
than 0.16.2 across short/31.7K greedy and sampled code prompts. Each cell had
four 512-token outputs; the older runtime ran first, and disk page cache was not
reset. This supports bounded throughput and compatibility gains, not general
performance at 262K.

## Validation commands and evidence

Run the HTTP screens against an otherwise idle active instance:

```bash
python3 tools/halogen_serving_validate.py --output /tmp/halogen-serving.json
python3 tools/halogen_performance_validate.py --rounds 2 --output /tmp/halogen-performance.json
# Requires an active NPU profile:
python3 tools/halogen_npu_validate.py --rounds 5 --output /tmp/halogen-npu.json
```

The checkpoint comparator temporarily switches models, then restores v2 with
NPU services. It needs both checkpoints and a prepared NPU host:

```bash
python3 tools/halogen_checkpoint_compare.py --output-dir /tmp/halogen-checkpoint-comparison
```

For dataset acquisition and NPU evaluation, see [Benchmarks](benchmarking.md#npu-retrieval-evaluation).
These screens do not run broader coding or long-context quality benchmarks.

| Evidence | Record |
| --- | --- |
| 0.17.3 runtime, serving, MTP, and NPU comparison | [Upgrade summary](results/halogen-0173-upgrade-2026-10-09/summary.json) |
| W4B/v2 memory, speed, quality, and retrieval | [Checkpoint comparison](results/halogen-0162-checkpoint-comparison-2026-10-05/comparison.json) |
| V2 GPU/NPU contention | [Paired measurements](results/halogen-0162-v2-npu-evaluation-2026-10-05/contention.json) |
| NPU ingestion bursts | [Request timings](results/halogen-0162-v2-npu-evaluation-2026-10-05/ingestion.json) |
| SciFact methods, rankings, and per-query scores | [Evaluation](results/halogen-0162-v2-npu-evaluation-2026-10-05/scifact.json) |
