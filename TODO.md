# Qwen3.8 ROCmFPX roadmap

The goal is to add Qwen3.8 support to Halo AI in the order that produces the
most value for the least downloading and engineering work.

## Decisions

- The first and only required baseline is the text-only
  `Qwen3.8-27B-ROCmFP4-FAST.gguf` running on the iGPU without NPU drafting.
- Built-in GPU MTP is the first acceleration to try because it uses the same
  model file and requires no draft-model download.
- External NPU drafting is an experiment for the FP4 target only. It proceeds
  only after a cheap acceptance and throughput screen.
- A production drafter must have Qwen3.8 lineage and the target tokenizer and
  semantics. Qwen3.6 may be used to test version mixing, never as the final
  drafter.
- ROCmFP8 is optional and GPU-only. There will be no FP8+NPU profile and no
  attempt to place the 27B FP8 model on the NPU.
- Vision is optional and separately downloaded. External NPU drafting remains
  disabled for image and video requests.
- Selecting one profile must never download artifacts belonging only to
  another profile or experiment.

## Stage 1: Ship the FP4 baseline

**Value:** a new, useful Qwen3.8 runtime with one model download and no NPU
dependency.

### Required artifact

- Repository:
  [`julianmb/Qwen-3.8-27B-ROCmFP4-FAST-GGUF`](https://huggingface.co/julianmb/Qwen-3.8-27B-ROCmFP4-FAST-GGUF/tree/e439c125804f73ccb1d9e6aebb5ea50f112833fa)
- Revision: `e439c125804f73ccb1d9e6aebb5ea50f112833fa`
- File: `Qwen3.8-27B-ROCmFP4-FAST.gguf`
- Size: `14562236384` bytes
- SHA-256: `fb89c78d2be91cdb68eaaaa45b1270710bf34aa721dc1f0b9e3aa7b98d2e1da9`
- Local filepath: `/srv/halo-ai/models/julianmb/Qwen-3.8-27B-ROCmFP4-FAST-GGUF`
- Preinstalled model; any approved sibling files from
  `julianmb/Qwen-3.8-27B-ROCmFP4-FAST-GGUF` should reside in that same directory.

### Tasks

- [ ] Pin the exact ROCmFPX/llama.cpp source commit required by q38rocm and
      build a reproducible, digest-pinned runtime image.
  The v1.0.0 archive, q38rocm release commit, and both base images are pinned,
  and the built image has a recorded digest. The binary reports `e87d53e`, but
  upstream does not expose a resolvable full source commit for that claim, so
  the source-provenance part of this item remains open.
- [x] Add a distinct `rocmfpx` engine to Halo; do not disguise the custom
      ROCmFPX ABI as ordinary `llamacpp`.
- [x] Add the FP4 artifact and a `qwen38-27b-rocmfp4-baseline` profile to the
      catalog with speculation and vision disabled.
- [x] Add profile-scoped acquisition with dry-run, resume, exact size/SHA
      verification, and atomic completion.
- [x] Make the acquisition dry-run prove that the baseline downloads no FP8,
      NPU model, visual projector, BF16 model, or reference quant.
- [x] Validate the GGUF header and tensor inventory. Trust the verified GGUF,
      not the repository's inconsistent companion `config.json`.
- [x] Add render, lifecycle, readiness, and deterministic smoke tests.
- [x] Benchmark unassisted greedy generation at short, 4K, and 32K contexts;
      record TTFT, prompt speed, decode speed, memory, and software revisions.
- [x] Document the baseline command, expected storage, supported text-only
      modality, and rollback procedure.

### Gate

- [x] **Pass:** Halo can acquire, start, test, stop, and reproduce the direct
      q38rocm FP4 result without downloading any optional model artifact.
- [ ] **Fail:** fix or stop here; do not begin MTP, NPU, FP8, or vision work.

## Stage 2: Enable built-in GPU MTP

**Value:** the largest likely speedup with zero additional model downloads.

### Tasks

- [x] Add `qwen38-27b-rocmfp4-mtp` using the MTP tensors already stored in the
      FP4 GGUF.
- [ ] Implement strict greedy MTP first and verify token-for-token equivalence
      with the unassisted FP4 profile.
  The backend's strict Qwen verifier is active and 4/5 stable corpus cases
  matched exactly. Full cross-process identity is not yet proven because the
  Vulkan baseline itself changes among semantically equivalent outputs across
  fresh processes despite fixed request and server seeds.
- [x] Benchmark unassisted FP4 versus strict greedy MTP on the same fixed
      prompts and contexts.
- [ ] Only after strict mode passes, evaluate q38rocm's suggested proposal
      settings such as `n_max=6`, `p_min=0.60`; label probabilistic modes
      clearly.
  The `n_max=6`, `p_min=0.60` settings were evaluated under the operator's
  explicit experimental-optimization goal, not promoted as a strict-mode pass.
- [x] Record drafted/accepted tokens, accepted-run distribution, latency,
      throughput, memory, and stability.

### Gate

- [ ] **Pass:** keep GPU MTP if it is correct, stable, and materially faster.
- [ ] **Fail:** retain the unassisted FP4 baseline and stop tuning MTP.

## Stage 3: Decide whether NPU drafting is worth building

**Value:** prevents a large download and substantial integration effort when
cross-version acceptance or NPU speed cannot beat GPU MTP.

No NPU model is downloaded at the start of this stage.

### Tasks

- [x] If a verified Qwen3.6-35B-A3B model is already present, reuse it for a
      no-new-download Qwen3.6-to-Qwen3.8 acceptance screen. If it is absent,
      skip this proxy rather than downloading another copy solely for it.
  The cached 39,099,447,584-byte GGUF and both templates passed full SHA-256
  verification. The short canary used them in place and downloaded zero model
  bytes; only the explicitly allowed Lemonade support runtime was refreshed.
- [x] Feed both models the exact Qwen3.8-rendered token prefix under greedy
      decoding; do not compare independently rendered chat prompts.
  Halo captured Qwen3.8 template/tokenizer output as token-ID arrays, extended
  each with the authoritative target suffix, and sent those arrays directly to
  Qwen3.6's private llama.cpp completion endpoint. All 70 prefix hashes matched.
- [ ] Test code, reasoning, JSON/tool output, multilingual text, ordinary chat,
      and thinking/non-thinking prompts at several context lengths.
  The short-context early screen covers every listed domain and both modes.
  Longer buckets remain unrun because thinking mode already crossed the
  permissive early-stop threshold; a bounded non-thinking-only context check
  remains optional.
- [ ] Measure first-token agreement, zero-acceptance rate, mean accepted tokens,
      accepted-run p50/p95, and results for proposal lengths 1, 2, 4, and 6.
  The short canary records every metric for all four proposal lengths, grouped
  by domain and mode. Non-thinking achieved 94.29% first-token agreement,
  5.71% zero acceptance, and 4.771/6 mean accepted tokens. Thinking achieved
  28.57%, 71.43%, and 1.143/6 respectively, so the general proxy is not being
  expanded. Several-context coverage remains open.
- [ ] Model expected end-to-end speed using measured FP4 verification time and
      realistic NPU proposal latency. Compare against FP4 GPU MTP, not merely
      unassisted FP4.
- [ ] Identify a genuinely trained small Qwen3.8-family draft model. Treat
      `inference-optimization/Qwen3.8-1.0B-A0.6B` only as a converter/kernel
      fixture because its card describes toy-data training.
- [ ] Confirm the candidate has exact tokenizer IDs, prompt/thinking semantics,
      credible training provenance, and useful quality before converting it.
- [ ] Only if the acceptance and candidate checks are promising, confirm that
      Vulkan and XDNA2 can coexist on the target Strix Halo host without resets
      or unacceptable contention.

### Gate

- [ ] **Proceed:** a credible small Qwen3.8 drafter exists, version/family
      acceptance is promising, coexistence is stable, and the throughput model
      predicts at least 10% over FP4 GPU MTP.
- [ ] **Stop:** ship Stages 1–2. Do not download Q4NX weights or build the
      speculative bridge.

## Stage 4: Build the minimum NPU prototype

This stage exists only if Stage 3 passes.

### Tasks

- [ ] Pin FastFlowLM, XRT, amdxdna, the draft model, converter, and matching
      XDNA `xclbin` kernels.
- [ ] Produce or acquire a Q4NX artifact from pinned SafeTensors. FastFlowLM
      does not directly load Unsloth iMatrix/IQ GGUF files.
- [ ] Download only the selected text drafter and required kernels; exclude
      FP8 and vision weights.
- [ ] Add the smallest useful token-level draft interface: session open,
      token-ID prefill, greedy propose, commit/rollback, reset/close, and health.
- [ ] Connect the FP4 verifier directly over a private Unix socket. Do not put
      Lemonade or per-proposal HTTP calls in the hot path.
- [ ] Keep all sampling and token emission authoritative in the FP4 verifier.
- [ ] Prove real XDNA execution with device counters and prohibit silent CPU or
      GPU drafting.
- [ ] Test full, partial, and zero proposal acceptance plus timeout, crash, and
      KV rollback behavior.
- [ ] Benchmark the complete system against FP4 GPU MTP on the Stage 2 corpus.
- [ ] Keep Qwen3.6 test-only even if it helps exercise the protocol; it cannot
      satisfy the production same-family requirement.

### Gate

- [ ] **Promote experimentally:** strict greedy output matches unassisted FP4,
      the soak test is clean, and median throughput is at least 10% above FP4
      GPU MTP without unacceptable p95 latency or power.
- [ ] **Reject:** retain the research code if useful, but do not expose an NPU
      profile in normal Halo operation.

## Stage 5: Optional capabilities

These tracks are independent, explicitly selected, and never prerequisites for
Stages 1–4.

### 5A: ROCmFP8 on the GPU

- [ ] Add the FP8 artifact as disabled, on-demand catalog metadata:
  - File: `Qwen3.8-27B-ROCmFP8.gguf`
  - Size: `28193396704` bytes
  - SHA-256: `0bf5bfc9f946090af2d41b388ccb4d627e916c7250517c36a0de37d6eaccfd8e`
- [ ] Require an explicit FP8 acquisition request and show its full byte cost
      before downloading.
- [ ] Add unassisted and built-in-MTP GPU profiles only; reject FP8+NPU catalog
      combinations.
- [ ] Compare FP8 with FP4 on a fixed task-quality and performance suite before
      describing it as effectively lossless or better quality.

### 5B: Vision with the FP4 target

- [ ] Keep the normal FP4 baseline text-only.
- [ ] On explicit vision acquisition, select one verified compatible visual
      companion rather than downloading every candidate.
- [ ] Validate projector provenance, preprocessing, special tokens, image
      handling, output quality, memory, and TTFT against the FP4 target.
- [ ] Use target-native execution/MTP for multimodal requests; keep external NPU
      drafting disabled for images and video.

## Known constraints

- The official
  [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)
  is multimodal, but the q38rocm FP4/FP8 GGUFs contain the language decoder and
  MTP tensors only. They do not contain the vision encoder.
- q38rocm's
  [`npu_sidecar_drafter.py`](https://github.com/julianmb/q38rocm/blob/ae3d4640041c92f4b0a012d6bed004e0ad9e66b9/scripts/npu_sidecar_drafter.py)
  detects the NPU but launches GPU MTP; it is not an NPU inference provider.
- FastFlowLM uses `model.q4nx` plus model-specific XDNA kernels, not arbitrary
  GGUF/iMatrix quantizations.
- FastFlowLM's Qwen3.6 support proves useful MoE NPU primitives exist, including
  prefill, forward, checkpoint, and restore. It does not prove that a differently
  sized Qwen3.8 model can reuse the same binary kernels.
- Qwen3.6 shares much of the tokenizer with Qwen3.8 but is a different model
  generation. Its proposal acceptance must be measured rather than inferred.
- FastFlowLM's public HTTP API does not currently provide the efficient,
  stateful token protocol required for speculative decoding.
- Lemonade can help with NPU detection and FastFlowLM installation/validation,
  but it does not currently supply the required draft protocol.

## Download budget

| Stage | Model download policy |
| --- | --- |
| 1 | FP4 GGUF only: 14,562,236,384 bytes |
| 2 | No additional model download |
| 3 | No additional model download by default; reuse verified cached artifacts |
| 4 | One selected Q4NX drafter plus required kernels, only after the Stage 3 gate |
| 5A | Optional FP8 GGUF: 28,193,396,704 bytes |
| 5B | One optional verified visual companion |

Do not download the full Unsloth quantization ladder, both visual companions,
BF16 reference weights, or the 23 GB Qwen3.6 Q4NX model unless a specific,
approved experiment requires that exact artifact.
