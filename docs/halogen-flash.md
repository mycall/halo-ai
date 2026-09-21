# Recommended Flash-Next: Halogen

[Halogen Flash Server](https://github.com/peonist-ai/halogen-flash-server)
0.12.2 is integrated as a separate, experimental Qwen3.8-Flash-Next runtime.
The engine is closed source; the public repository contains its deployment
surface. The image is pinned to
`sha256:8f4c75fc15a0e2f2c023241389c4c946a60fa4fb4cd6d350dfcea17e22d119f9`.

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

## Run from this checkout

Until the system installation is reloaded, use the repository CLI and its
workspace configuration:

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
