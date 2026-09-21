# Halogen Flash candidate

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
| `qwen3.8-halogen` | 262,144 | MTP plus prompt lookup | Session reuse (mode 2) | On |

`qwen3.8-halogen` aliases `qwen3.8-flash-next-halogen-262k-vision`.
The existing recommended Qwen aliases are unchanged. The serving profile
uses one slot, a 262,144-position shared KV pool, and 16,384-token prefill
chunks. The smaller prefill arena leaves more room for the desktop. Context
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
./bin/halo-ai --config .halo-ai-workspace.env start qwen3.8-halogen --switch
./bin/halo-ai --config .halo-ai-workspace.env test qwen3.8-halogen
./bin/halo-ai --config .halo-ai-workspace.env stop qwen3.8-halogen
```

The OpenAI-compatible endpoint is `http://127.0.0.1:8731/v1`; the served model
ID is `halogen-qwen3.8-flash-next`. Chat Completions and Responses are both
available. `/health` reports the engine and API versions, MTP readiness,
vision support, and the verified chat template. Only the API port is
published, on loopback. The internal engine port remains private.

Model mounts are read-only, downloads are disabled in the container, and
Halo's normal single-runtime switching, stop, status, and trial records apply.
Starting the candidate is explicit; it does not start automatically at boot.

To update the root-owned system CLI and catalog after reviewing the changes,
run `./reload.sh` from this checkout (requires your sudo password). Then the
same commands work as `halo-ai start qwen3.8-halogen --switch`, etc.

## Verify MTP

```bash
./bin/halo-ai --config .halo-ai-workspace.env start qwen3.8-flash-next-halogen-32k-mtp --switch
python3 tools/halogen_validate.py --output /tmp/halogen-mtp-check.json
./bin/halo-ai --config .halo-ai-workspace.env start qwen3.8-halogen --switch
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

**Recommendation: keep Halogen as an experimental option.** MTP is active,
and the serving integration works, but these results do not support promoting
it to the general Qwen route or claiming reliable retrieval across the entire
262K context. The smaller-context successes are bounded smoke checks, not a
broad quality qualification. Existing recommended aliases remain unchanged.

[Full serving results and retry records](results/halogen-flash-native-serving-2026-09-20.json)
include the request policies, prompt hashes, timing/cache counters, and outputs.
