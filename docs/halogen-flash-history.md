# Halogen test and setup history

> Historical record through 2026-10-05. For the current conclusion, profile
> selection, and test limitations, read [the current runbook](halogen-flash.md).
> Statements such as “pending reboot” describe their dated stage of setup;
> they are not instructions to repeat those changes on the current host.

## Original integration notes

[Halogen Flash Server](https://github.com/peonist-ai/halogen-flash-server)
0.16.2 is integrated as a separate, experimental Qwen3.8-Flash-Next runtime.
The engine is closed source; the public repository contains its deployment
surface. The image is pinned to
`sha256:0c61bf84ac22308a53f5d1ca6b86806702d7039e5ebc51cae4c66621b92fe04a`.

The catalog reuses the files under
`/srv/halo-ai/models/peonist-ai/halogen-qwen3.8-flash-next`, pinned to
Hugging Face revision `e238ffde1a89531743505349dd16ca35a1e09b15`.
It verifies HGN headers, expected sizes, tokenizer files, and full SHA-256
hashes through `models verify --full`. The checkpoint and **quality** overlay
are both required. The speed overlay is cataloged for verification but is
not selected. The native checkpoint contains its own MTP head; the separate
`qwen38-flash-next-mtp.hgn` is for upstream's GGUF loading path and is not
mounted by these native profiles.

| Profile | Context | Drafting | Prompt cache | Vision |
| --- | ---: | --- | --- | --- |
| `qwen3.8-flash-next-halogen-32k-baseline` | 32,768 | Serial, prompt lookup off | Off | Off |
| `qwen3.8-flash-next-halogen-32k-mtp` | 32,768 | MTP only, prompt lookup off | Off | Off |
| `qwen3.8-fn` / `qwen3.8-halogen` | 262,144 | MTP plus prompt lookup | Session reuse (mode 2) | On |

`qwen3.8-fn` is the recommended Flash-Next alias and resolves to
`qwen3.8-flash-next-halogen-262k-vision`; `qwen3.8-halogen` remains an alias
for the same profile. This selects the native HGN model, not UD-Q4_K_XL.
The recommendation reflects the local performance and correctness comparisons
below. The profile retains its experimental risk label: neither runtime
reliably retrieved all three codes near 262K, including with MTP disabled.

The serving profile uses one slot, a 262,144-position shared KV pool, and
16,384-token prefill chunks. The smaller prefill arena leaves more room for the desktop. Context
size and prefill chunk size are separate settings. The startup log reports
actual locked weights and remaining host memory; `MemAvailable` includes
locked file pages and overstates what another process can use.

The request defaults are medium reasoning, an 8,192-token completion budget,
and temperature 1.0 / top-p 0.95 / top-k 20. Clients can override them;
`temperature: 0` explicitly selects greedy decoding. Thinking consumes part
of the completion budget. Mode 2 cache reuse is intended for interactive
serving and is not guaranteed byte-identical to a cold request; use the
32K profiles for the integration comparison.

## 0.16.2 upgrade and optional profiles (2026-10-04)

The default aliases still use the verified W4B checkpoint and quality overlay.
The engine upgrade does not download or select v2. Historical measurements below
are from 0.12.2; new results are recorded separately. The public upstream tree
contains deployment code and release notes, not the closed-source engine.

The same-process serial/MTP check on 0.16.2 passed all text-equality checks:
13/13 suite outputs plus the 512-token probe matched. Both scored 10/13, with
unchanged code-trace, code-slice and CRT failures. The probe measured 31.797
tok/s serial and 55.489 tok/s MTP; these are single observations, not a repeated
speed comparison with 0.12.2. All serial requests drafted zero tokens; MTP
accepted 388 of 416 proposals across the check. The API regression screen
passed streaming role/text, streamed tool calls, split assistant/tool replay,
structured JSON, Anthropic Messages/token counts, and an image in a Responses
tool result. The normal red-image CLI smoke also passed.

- [MTP results](results/halogen-0162-mtp-2026-10-04.json)
- [Serving regression results](results/halogen-0162-serving-2026-10-04.json)
- [Release review](halogen-upstream-review.md)
- [Serial retrieval retest](results/halogen-0162-retrieval-2026-10-04.json)
- [NPU host preparation](results/halogen-0162-host-preparation-2026-10-04.json)

The exact September retrieval prompts were repeated: 261,107 tokens returned
2/3 codes again (wrong first code `delta-90571`), while the 31,724-token control
returned 3/3. Both drafted zero tokens and reused zero cached tokens. Their
message hashes match the original tests. Request times were 246.34 s and
31.66 s; host setup ran during this screen, so these are diagnostic timings,
not a controlled performance comparison. The upgrade does not fix this
near-limit retrieval failure.

All 161 unit tests and the repository smoke script passed. The NPU host change
was staged with boot backup `20261004T235133Z` and Snapper pair 137/138. Running
mode remains GPU; persistent mode is NPU with the same 118 GiB GTT setting.
The fabric-clock service is enabled for the next boot/resume. A manual reboot
was required before actual NPU validation; the completed post-reboot checks are
recorded below. We did not run gaming benchmarks;
removing `amd_iommu=off` is for NPU compatibility, not a measured gaming gain.


### Download v2 separately

The catalog pins Hugging Face revision
`3648cf1e6a3143e8946d52301570f26003c0bdb0`. Download both files into
`/srv/halo-ai/models/peonist-ai/halogen-qwen3.8-flash-next/`:

| File | Bytes | Direct download |
| --- | ---: | --- |
| `qwen38-flash-next-v2.hgn` | 66,687,678,432 | [Download v2](https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next/resolve/3648cf1e6a3143e8946d52301570f26003c0bdb0/qwen38-flash-next-v2.hgn?download=true) |
| `qwen38-flash-next-ngram.hgn` | 51,200,246,144 | [Download lookup table](https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next/resolve/3648cf1e6a3143e8946d52301570f26003c0bdb0/qwen38-flash-next-ngram.hgn?download=true) |

The tokenizer and vision files are byte-identical to our existing pinned files
and can be reused. V2 mounts its separate n-gram table and does not mount the
W4B overlay or the external GGUF MTP head. Its catalog checksums are in
`config/models.d/halogen.json` and the main runbook's manifest. Both files are
downloaded and passed full SHA-256 verification on 2026-10-05 UTC, along with
the reused tokenizer and vision files. Verification evidence is in
`results/halogen-0162-v2-verification-2026-10-05.json`.

The requested switch to `qwen3.8-halogen-v2-npu` succeeded on 2026-10-05.
The running 0.16.2 server reports the v2 model, a 262,144-token context, and
healthy GPU and NPU capabilities. The vision smoke returned `red`; all eight
NPU correctness checks passed, including a concurrent 512-token GPU generation
with 11 embedding and 11 reranking requests. NPU outputs matched their idle
references exactly. Evidence is in
`results/halogen-0162-v2-smoke-2026-10-05.json` and
`results/halogen-0162-v2-npu-switch-2026-10-05.json`.
These switch checks were followed by the controlled comparison below.
Extended stability and broad model-quality qualification remain pending.

After downloading:

```bash
halo-ai models verify halogen-qwen3.8-flash-next-v2 --full
halo-ai start qwen3.8-flash-next-halogen-v2-32k-mtp --switch
python3 /opt/halo-ai/current/tools/halogen_validate.py --output /tmp/halogen-v2-mtp.json
halo-ai start qwen3.8-halogen-v2 --switch
halo-ai test qwen3.8-halogen-v2
```

The corresponding baseline is `qwen3.8-flash-next-halogen-v2-32k-baseline`.
The existing `qwen3.8-fn` alias remains the W4B profile; switching the running
server does not change aliases.

### W4B versus v2 comparison (2026-10-05)

The fresh comparison used the same pinned 0.16.2 image and current host for both
checkpoints, with NPU and vision disabled during measurement. Cache and prompt
lookup were off; requests used temperature 0, seed 1, and thinking off. The
existing 13-case suite ran with serial and MTP decoding, followed by three
512-token generation runs per mode at 32k context. Serial/MTP order alternated
within each checkpoint. W4B ran first, then v2; model order was not randomized.

| Measurement | W4B + quality overlay | V2 |
| --- | ---: | ---: |
| Fixed correctness suite, serial and MTP | 10/13 | 11/13 |
| Serial decode, median of three | 35.562 tok/s | 35.595 tok/s |
| MTP decode, median of three | 58.596 tok/s | 57.469 tok/s |
| Engine-reported locked weights | 68.0 GiB | 62.1 GiB |
| Engine-reported total at 32k context | 79.5 GiB | 70.8 GiB |
| Retrieval at 31,724 prompt tokens | 3/3 codes | 3/3 codes |
| Retrieval at 261,107 prompt tokens | 2/3 codes | 2/3 codes |

V2 corrected the modular-arithmetic case (`38`); both code-tracing/slicing
questions still failed. Within each checkpoint, all 13 cases and the generation
probe matched between serial and MTP, and the repeated probe outputs were
stable. The generation text differs across checkpoints, so the speed figures
compare the same prompt and output-token budget, not an identical token stream.

There was no meaningful decode-speed improvement in this sample: v2 serial
was +0.09% and MTP was -1.92%. Run ranges overlapped (W4B serial 32.088–35.739,
v2 serial 30.337–35.672; W4B MTP 57.830–58.830, v2 MTP 52.177–57.900 tok/s).
The first measured run was slower for both models, despite the preceding suite;
these small differences should not be treated as a robust throughput ranking.
V2 saved 5.9 GiB of locked weights and 2.8 GiB of working memory, totaling
8.7 GiB less engine-held memory under the matched 32k setup. These engine
figures exclude the demand-paged lookup table and other host processes.

The retrieval prompts were identical across checkpoints. Both still missed
the first near-limit code, although the wrong answer changed. Near-limit
prefill took 273.25 s with W4B and 266.29 s with v2; the shorter control took
33.42 s and 32.79 s respectively. Each length ran once per checkpoint, so these
timings are observations rather than a reliable prefill-speed improvement.

The run restored `qwen3.8-halogen-v2-npu` with the 262k context. The restored
vision smoke, NPU embeddings, and reranking passed. No GPU/NPU/IOMMU/OOM or
hung-task kernel messages matched during this bounded run. The ordinary
`qwen3.8-fn` alias still points to W4B; this comparison did not change aliases.

Evidence: [comparison and effective profiles](results/halogen-0162-checkpoint-comparison-2026-10-05/comparison.json),
[W4B correctness](results/halogen-0162-checkpoint-comparison-2026-10-05/w4b-quality.json),
[v2 correctness](results/halogen-0162-checkpoint-comparison-2026-10-05/v2-quality.json),
[W4B retrieval](results/halogen-0162-checkpoint-comparison-2026-10-05/w4b-retrieval.json),
[v2 retrieval](results/halogen-0162-checkpoint-comparison-2026-10-05/v2-retrieval.json).
The evidence directory also includes startup memory logs and restored-service
checks. Reproduce from this checkout (temporarily switches the running model):

```bash
python3 tools/halogen_checkpoint_compare.py --output-dir /tmp/halogen-checkpoint-comparison
```

### Optional NPU embeddings and reranking

`qwen3.8-halogen-npu` adds `qwen3-embedding-0.6b` and
`qwen3-reranker-0.6b` to the W4B serving profile on the same port.
`qwen3.8-halogen-v2-npu` does the same with v2. Neither is the default.
The model list is explicit in each profile. The pinned manifest supports
upstream's other three small models too, but those are not selected/downloaded
by the shipped profiles and have not been locally tested.

The library manifest was extracted from the digest-pinned 0.16.2 image's
`/opt/halogen/npu/models.txt`; it records upstream revisions, sizes and hashes.
The preparation helper fetches only selected files and shared device programs,
verifies their SHA-256, and stores them under
`/srv/halo-ai/models/halogen-npu/0.16.2`. All model and XRT mounts are read-only;
serving does not enable downloads, bypass fabric-clock checks, or mount writable
host sysfs. Startup requires a successful engine capability probe and every
selected NPU model to appear in `/v1/models`.

```bash
python3 /opt/halo-ai/current/tools/halogen_npu_prepare.py --download
# Host changes: review the dry run, then use KDE authentication.
bash /opt/halo-ai/current/tools/halogen_npu_host_setup.sh --dry-run
pkexec /usr/bin/bash /opt/halo-ai/current/tools/halogen_npu_host_setup.sh
# Reboot manually with IOMMU enabled in firmware, then:
halo-ai host-profile status
halogen-fabric-clock status
python3 /opt/halo-ai/current/tools/halogen_npu_prepare.py
halo-ai start qwen3.8-halogen-npu --switch
halo-ai test qwen3.8-halogen-npu
```

The host helper installs CachyOS's `xrt` and `xrt-plugin-amdxdna`, downloads
checksum-pinned upstream fabric-clock files, enables that service for the next
boot/resume, and stages the existing conservative `npu` boot profile with a
snapshot and verified boot backup. It does not reboot or change GTT/TTM.
This profile removes `amd_iommu=off` and uses the kernel's translated IOMMU
default; it does not introduce `iommu=pt`. The GPU may have different performance
after this change, so current GPU-only timings do not qualify GPU+NPU serving.
The service stays enabled at subsequent boots, even with a GPU-only profile;
to undo that part, disable it and run `halogen-fabric-clock release` as root.

XRT is mounted by resolving each library's real file and exposing it at both
`/opt/xilinx/xrt/lib` and the host library directory. All three CachyOS XRT
libraries successfully loaded in the 0.16.2 image during preparation. Set
`HALOGEN_XRT_LIB_DIR` if the host uses a directory other than `/usr/lib`.
The embedding and reranker files passed full hashes. Post-reboot NPU inference
and bounded concurrent GPU/NPU checks subsequently passed, as recorded below.

The embedding route is `/v1/embeddings`; its 1024-dimensional vectors are
normalized. Retrieval queries need the model's `Instruct: ...\nQuery:...`
format. Reranking uses `/v1/rerank`; the 4096-token limit includes both query
and document/template. See [upstream request shapes](https://github.com/peonist-ai/halogen-flash-server/blob/7f31bbd4021f217a1be9776bdb7304bcf8eca62d/docs/NPU.md).

### Post-reboot NPU validation (2026-10-04)

The running and persistent host profiles now both report `npu`, with no reboot
pending, the same 118 GiB GTT aperture, kernel `7.2.3-1-cachyos-deckify`, and
`amdxdna` bound. The fabric-clock service succeeded at boot and held FCLK at
2000 MHz. Its `inactive (dead)` state after a successful run is normal for this
oneshot service; `halogen-fabric-clock status` verifies the actual clock.

The first start exposed a missing host prerequisite: the rootless login had an
8 MiB hard memlock limit, while upstream required about 2.1 GiB for the two NPU
models. Container `--ulimit` alone cannot lift the operator's hard limit. The
new `tools/halogen_memlock_setup.py` configures the named operator's PAM limits,
their `user@UID.service` limit, and their user-manager default limit. It can
also raise an explicitly selected existing operator process and its future
children, avoiding another reboot. The preparation helper and profile startup
now check the hard limit before attempting a container launch. Startup health
also requires `npu.status=ok` and both selected models in the NPU health record
and `/v1/models`.

```bash
# If an existing terminal still inherited the old limit, use a renewed login.
ulimit -H -l
# Persistent setup is now included in halogen_npu_host_setup.sh; independently:
pkexec /usr/bin/python3 /opt/halo-ai/current/tools/halogen_memlock_setup.py --user "$USER"
```

Both endpoints passed shape, normalization, semantic ordering and repeat checks.
Batched embeddings exactly matched individual requests; 128-dimensional vectors
matched renormalized prefixes, and base64 matched float output. Reranking sorted
relevant documents first and handled `top_n` and document objects. Both routes
refused oversized inputs with HTTP 400. Sixteen simultaneous embedding requests,
each with eight inputs, all succeeded and returned the expected vectors.

Three alternating pairs of 512-token GPU decode probes measured medians of
54.131 tok/s with the NPU idle and 39.875 tok/s with continuous embedding and
reranking traffic (about 26% slower). The six GPU outputs were byte-identical.
A separate concurrent correctness check ran ten embedding requests and ten
reranking requests alongside GPU generation: both vectors and scores were
identical to their idle references, with zero maximum absolute difference.
Post-load correctness checks also passed. The kernel journal showed no new
XDNA/IOMMU faults or GPU reset/error/timeout reports during these checks.

Isolated NPU throughput used one warmup plus three measured requests per case,
with the GPU model loaded but idle. Each text repeats a simple fixture sentence;
these are synthetic observations, not a reproduction of upstream's corpus.
Token counts come from API usage and include the rerank template/query.

| Endpoint | Tokens per input/pair | Batch | Median throughput |
| --- | ---: | ---: | ---: |
| Embeddings | 502 | 8 | 9,232 tokens/s |
| Embeddings | 522 | 8 | 3,430 tokens/s |
| Reranking | 499 | 8 | 18.74 pairs/s |
| Reranking | 599 | 8 | 6.59 pairs/s |
| Embeddings | 502 | 1 | 99.2 ms/request |

Crossing from short inputs to the next size is costly in these measurements.
Keep the complete embedding input or rerank pair within the short-input budget
when batching for throughput; a document's token count alone omits query/template
overhead. These figures are not interchangeable with Flash generation tok/s.

The five GPU serving regression checks also passed with both NPU models loaded:
streaming text/tools, split assistant/tool replay, structured JSON, Anthropic
Messages and token counting, and Responses image tool results. All 162 unit
tests and the repository smoke script passed after the prerequisite fixes.
No gaming benchmark, suspend/resume cycle, or extended stability soak was run.
V2 subsequently passed the switch checks and bounded checkpoint comparison
recorded above.

Reproduce (run the benchmark without other inference clients):

```bash
halo-ai start qwen3.8-halogen-npu --switch
python3 /opt/halo-ai/current/tools/halogen_npu_validate.py --output /tmp/halogen-npu-check.json
python3 /opt/halo-ai/current/tools/halogen_npu_bench.py --output /tmp/halogen-npu-bench.json
```

Evidence: [host and memlock](results/halogen-0162-npu-host-2026-10-04.json),
[API and paired load](results/halogen-0162-npu-2026-10-04.json),
[concurrent reference values](results/halogen-0162-npu-reference-2026-10-04.json),
[isolated throughput](results/halogen-0162-npu-throughput-2026-10-04.json),
[GPU serving with NPU loaded](results/halogen-0162-serving-npu-2026-10-04.json).

## Run from this checkout

For development, use the repository CLI and its workspace configuration:

```bash
./bin/halo-ai --config .halo-ai-workspace.env install halogen
./bin/halo-ai --config .halo-ai-workspace.env models verify halogen-qwen3.8-flash-next --full
./bin/halo-ai --config .halo-ai-workspace.env start qwen3.8-fn --switch
./bin/halo-ai --config .halo-ai-workspace.env test qwen3.8-fn
./bin/halo-ai --config .halo-ai-workspace.env stop qwen3.8-fn
```

The OpenAI-compatible endpoint is `http://127.0.0.1:8731/v1`; the served model
ID is `halogen-qwen3.8-flash-next`. Chat Completions and Responses are both
available. `/health` reports the engine and API versions, MTP readiness,
vision support, and the verified chat template. Only the API port is
published, on loopback. The internal engine port remains private.

Model mounts are read-only, downloads are disabled in the container, and
Halo's normal single-runtime switching, stop, status, and trial records apply.
Starting the runtime is explicit; it does not start automatically at boot.

To update the root-owned system CLI and catalog after reviewing the changes,
run `./reload.sh` from this checkout (requires your sudo password). Then the
same commands work as `halo-ai start qwen3.8-fn --switch`, etc.

## Verify MTP

```bash
./bin/halo-ai --config .halo-ai-workspace.env start qwen3.8-flash-next-halogen-32k-mtp --switch
python3 tools/halogen_validate.py --output /tmp/halogen-mtp-check.json
./bin/halo-ai --config .halo-ai-workspace.env start qwen3.8-fn --switch
```

The validator requires cache mode 0 and prompt lookup off to isolate MTP.
It compares serial and MTP greedy responses in the same process, using a
512-token code probe and the repository's 13-case quality suite. The code
probe omits `drafter` for the candidate request to exercise the configured
MTP default. It records `draft_n`, `draft_n_accepted`, timings, content hashes,
and quality scores. Serial requests must have zero drafts, MTP must have
accepted drafts, and the returned content must match. These checks compare
returned text bytes, not hidden engine token IDs, and do not constitute a
broad quality or long-context qualification.

Upstream's `shortlist_draft_head: false` health field describes a different
optimization, not MTP availability. Use `drafter_weights_loaded`,
`drafters_available`, and actual request counters to determine whether MTP
is active. See the [upstream flags](https://github.com/peonist-ai/halogen-flash-server/blob/main/docs/FLAGS.md).

## Local setup result (2026-09-20)

All 11 cataloged model/tokenizer files passed full SHA-256 verification.
The cache-off, prompt-lookup-off MTP check accepted 276 of 311 draft tokens;
serial requests drafted zero. All 13 fixed-suite outputs and the 512-token
code probe matched byte-for-byte between serial and MTP. Both scored 10/13;
the code-trace, code-slice, and CRT cases failed in both modes.

The single code probe decoded at 32.796 tok/s serial and 46.075 tok/s MTP
(40.5% higher throughput). This is a same-process, short-prompt observation,
not a throughput qualification across workloads. The full-context vision
profile then passed the red-image smoke, Responses API, and structured JSON
checks. Its startup reported 68.0 GiB pinned weights, 7.2 GiB KV, and 12.2 GiB
working memory (87.4 GiB total), leaving 21.6 GiB for the host at startup.
The subsequent near-limit retrieval checks below exercised the loaded 262K
profile and found an accuracy limitation.

See [the recorded results](results/halogen-flash-mtp-integration-2026-09-20.json).

## Post-reload serving and retrieval checks

The installer now deploys and verifies every shipped catalog, including
Halogen, and includes the MTP validator in the installed release. The system
CLI recognizes `qwen3.8-halogen`; streaming chat, a tool-call/result round
trip, and the configured medium-reasoning default passed live checks.

Three-code retrieval produced the following results. The 32K and 128K
screens use targets at 10%, 50%, and 90% of their respective documents. All
four near-limit attempts target the same three records and expected codes.

| Attempt | Prompt tokens | Cached prompt tokens | Reasoning | Correct codes |
| --- | ---: | ---: | --- | ---: |
| 32K screen | 31,724 | 0 | Off | 3/3 |
| 128K screen | 130,031 | 0 | Off | 3/3 |
| Near-limit, prefix reuse | 261,107 | 129,984 | Off | 2/3 |
| Identical prompt, fresh process | 261,107 | 0 | Off | 2/3 |
| Identical prompt, warm repeat | 261,107 | 261,107 | Off | 2/3 |
| Same targets, trailing distractors removed | 253,944 | 0 | Medium | 2/3 |

The medium retry reserved 8,192 output tokens and capped thinking at 2,048;
it used 707 reasoning tokens and finished normally. Every near-limit attempt
missed the first requested record and retrieved the other two correctly.
Changing both the reasoning policy and the prompt length did not resolve the
miss in this test. The cached and fresh attempts returned different wrong
first codes, so cache reuse alone does not explain the failure.

The requested first record was `000791`, whose code is `delta-68658`.
The wrong codes correspond to other records in the supplied document:
`delta-92415` belongs to `000794`, `delta-16172` to `000797`, and
`delta-30330` to `000079`. These were incorrect record selections rather
than codes absent from the source.

These initial results keep the 262K serving profile experimental. They do not
establish that UD-Q4_K_XL is more accurate at the same length; the matched
serial comparison below addresses that question. The smaller-context
successes are bounded smoke checks, not a broad quality qualification.

[Full serving results and retry records](results/halogen-flash-native-serving-2026-09-20.json)
include the request policies, prompt hashes, timing/cache counters, and outputs.

## Serial retest against UD-Q4_K_XL

A fresh process for each runtime received the identical 261,107-token
three-code request with greedy decoding, seed 1, thinking off, and a
1,024-token answer budget. Both used 262,144 context positions and one slot.
Halogen used serial decoding with prompt lookup and prompt caching disabled.
The UD-Q4_K_XL model used Strix Vulkan 0.7.5 with `--spec-type none`, no
draft-model mount, and `cache_prompt: false`; its live slot reported
`speculative: false`. Both processed the full prompt with zero cached tokens
and finished normally after 24 completion tokens.

| Model/runtime, MTP off | First code returned | Correct codes | Request time |
| --- | --- | ---: | ---: |
| Halogen native HGN | `delta-92415` (record `000794`) | 2/3 | 279.74 s |
| UD-Q4_K_XL / Strix Vulkan 0.7.5 | `delta-33577` (record `003792`) | 2/3 | 1,742.46 s |

The expected first code was `delta-68658` from record `000791`. Both returned
the correct middle and final codes. Halogen's answer matched its earlier
cold MTP run exactly, despite drafting zero tokens. The UD runtime does not
expose draft counters in these responses; disabling speculation is verified
by its command line and active slot, rather than treating absent counters
as measured zero.

**Disabling MTP did not fix this retrieval failure in either model/runtime
pair.** This does not establish the underlying cause: model conversion,
quantization, KV representation, and runtime differ between the two pairs.

Both serial runs subsequently retrieved all three codes from the 31,724-token
control and scored **10/13** on the fixed quality suite. Both failed
`code-trace-alternating`, `code-slice-semantics`, and `reasoning-crt`.
Halogen's 13 answers matched its earlier serial and MTP answers byte for byte;
UD's 13 answers matched its earlier target-only baseline. All 15 Halogen
requests drafted zero tokens, and all 30 requests across both runtimes used
zero cached tokens.

**Halogen is the recommended Flash-Next runtime through `qwen3.8-fn`**, with
UD target-only as a reference/fallback. These screens show no accuracy
advantage for UD, and Halogen processes the long prompts faster. The alias
selects the existing 262K serving profile with MTP and vision; the 32K
profiles remain diagnostics. Neither pair is qualified for reliable retrieval
near 262K. The recommendation does not establish broad model-quality
superiority, and selecting the alias does not change the tested runtime settings.

[Full serial retest results](results/qwen3.8-flash-next-serial-retest-2026-09-20.json)
include both effective diagnostic profiles, image pins, model artifacts,
request options, prompt hashes, outputs, timings, and speculation evidence.

The [upstream review](halogen-upstream-review.md) follows the author's recent
Reddit discussions into the dated issue resolutions and distinguishes
attention-budget experiments from already-fixed bugs and speed tuning.
