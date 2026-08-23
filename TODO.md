# Qwen3.8 ROCmFPX roadmap

The goal is to add Qwen3.8 support to Halo AI in the order that produces the
most value for the least downloading and engineering work.

## Decisions

- The first and only required baseline is the text-only
  `Qwen3.8-27B-ROCmFP4-FAST.gguf` running on the iGPU without NPU drafting.
- Built-in GPU MTP was the first acceleration tried because it uses the same
  model file and requires no draft-model download. It is now gated: the fixed
  quality suite disproved strict output identity for every tested policy.
- External NPU and cross-version drafting are deferred; their research notes
  remain in `docs/TODO backup.md`.
- The already-present ROCmFP8 artifact is optional and GPU-only. There will be
  no FP8+NPU profile.
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
- [x] **Not taken:** the baseline gate passed, so the mutually exclusive failure
      branch did not apply.

## Stage 2: Enable built-in GPU MTP

**Value:** the largest likely speedup with zero additional model downloads.

### Tasks

- [x] Add `qwen38-27b-rocmfp4-mtp` using the MTP tensors already stored in the
      FP4 GGUF.
- [x] Run strict greedy MTP against the unassisted FP4 profile and verify
      token-for-token equivalence across clean processes.
  **Result: failed.** The FP4 baseline was stable on all 13 cases across two
  fresh processes. Aggressive MTP matched only 7/13 output-token sequences;
  conservative MTP matched 11/13. The strict-Qwen backend flag therefore does
  not establish end-to-end greedy identity on this build.
- [x] Benchmark unassisted FP4 versus strict greedy MTP on the same fixed
      prompts and contexts.
- [x] Resolve the conditional evaluation of q38rocm's suggested proposal
      settings such as `n_max=6`, `p_min=0.60`; label probabilistic modes
      clearly.
  The `n_max=6`, `p_min=0.60` settings were evaluated under the operator's
  explicit experimental-optimization goal, not promoted as a strict-mode pass.
- [x] Record drafted/accepted tokens, accepted-run distribution, latency,
      throughput, memory, and stability.

### Gate

- [x] **Not taken:** the mutually exclusive GPU MTP pass branch did not apply.
- [x] **Fail:** retain the unassisted FP4 baseline and stop tuning MTP. All four
      ROCmFPX MTP profiles remain cataloged but are gated pending a specific
      backend fix and a complete rerun of the identity suite.

## Stage 3: Optimize same-host long-context PP/TPS

**Value:** improve the working Qwen3.8 GPU profiles before adding another
runtime architecture or model family.

### Current measured state

The controlled 2026-08-18 office run uses exact 4,095- and 31,998-token
non-repeating arrays, 64 generated tokens, three repetitions, container-local
requests, and 50 ms memory sampling. The table reports medians and the maximum
sampled GTT; the machine-readable record retains each repetition and its
variability.

| Profile | 4K PP / TPS | 32K PP / TPS | Peak GTT at 32K |
| --- | ---: | ---: | ---: |
| FP4 baseline | 180.86 / 12.89 | 123.75 / 10.78 | 14.11 GiB |
| FP4 MTP, F16 draft K/V, 6 / 0.60 | 169.94 / 21.71 | 120.96 / 17.85 | 19.09 GiB |
| FP4 MTP, q5_1 draft K/V, 6 / 0.60 | 170.41 / 21.86 | 122.59 / 18.13 | 19.00 GiB |
| FP4 MTP, q5_1 draft K/V, 2 / 0.85 | 170.69 / 19.58 | 104.03 / 13.75 | 18.43 GiB |
| FP8 baseline | 177.17 / 7.64 | 120.55 / 6.90 | 26.33 GiB |
| FP8 MTP | 166.68 / 19.86 | 117.35 / 14.95 | 31.14 GiB |

The aggressive q5_1 candidate was effectively throughput-neutral versus its F16
draft-cache control while saving 0.09--0.11 GiB GTT. Against unassisted FP4 it
improves median decode throughput by 69.61% at 4K and 68.25% at 32K, with an
end-to-end crossover near 44 and 66 generated tokens respectively, at a
4.89--4.91 GiB GTT cost. The conservative policy saves another 0.56--0.57 GiB
but loses 10.41% and 24.16% decode throughput. The subsequent quality gate
disqualified every MTP profile despite these speed results. FP8 uses
substantially more GTT and remains a quality experiment, not a performance
default.

### Tasks

- [x] Add a container-only exact-token ROCmFPX context benchmark with cold-cache
      enforcement, PP/TPS/TTFT, MTP acceptance, and peak-memory reporting.
- [x] Add a comparison gate that verifies identical prompt-token hashes and
      reports performance deltas and end-to-end crossover lengths.
- [x] Catalog and semantically verify the already-present FP8 artifact; its
      size and full SHA-256 match the pinned metadata and acquisition requires
      zero additional bytes.
- [x] Add explicit GPU-only FP8 baseline and MTP profiles.
- [x] Compare FP4 baseline/MTP and FP8 baseline/MTP on identical 4K/32K token
      arrays without using performance claims from another host.
- [x] Test `ngram-mod,draft-mtp` with match/min/max `24/48/64`. Build 213 parses
      the settings but refuses startup because ngram-mod disables recurrent
      rollback while strict Qwen MTP requires rollback covering the full draft.
      Do not expose this non-starting combination as a profile.
- [x] Repeat the FP4 baseline and FP4 MTP finalists at least three times using
      the non-repeating prompt pattern; report median and variability.
- [x] Add a single long-run office matrix that executes the source tests and
      every selected runtime sequentially, captures power/environment state,
      preserves per-profile logs, stops containers between profiles, and uses
      measured elapsed history to order the next run fastest-to-slowest.
- [x] Run that matrix with USB power and the office workload held stable. The
      clean run held both power profiles at `performance`, observed 45.049 W
      peak AMDGPU PPT, detected no suspend, passed 82 containerized source tests,
      and produced 23 valid profile results. One DS4 row was invalidated after a
      runner bug was found; one standalone vision profile was deliberately
      excluded because it was the active starting trial at the prior reboot.
- [ ] Audit and tune only locally supported batch, micro-batch, thread-batch,
      KV-cache, and execution options, one axis at a time.
- [x] Measure an otherwise-identical FP4 MTP candidate with the draft K/V cache
      compressed from its current implicit F16 default to `q5_1`. Record whether
      this recovers a useful portion of MTP's roughly 5.2 GiB GTT overhead
      without hurting acceptance or end-to-end throughput. It recovered only
      0.09--0.11 GiB, but median PP/TPS and acceptance remained effectively
      neutral or slightly better.
- [x] Compare the current aggressive `n_max=6`, `p_min=0.60` proposal policy
      with the conservative `n_max=2`, `p_min=0.85` field-report policy after
      draft-cache compression is held constant. Treat the other-host report as
      a hypothesis only and retain same-host acceptance/crossover evidence. The
      conservative policy saved 0.56--0.57 GiB more GTT but lost 10.41% decode
      TPS at 4K and 24.16% at 32K. The later correctness gate supersedes this
      performance-only ranking.
- [ ] With MTP gated, isolate batch `2048/1024` versus `1024/512`,
      `threads-batch=32`, and `fit=off` on the FP4 baseline only; do not combine
      axes in the first screen.
- [x] Run a fixed FP4-versus-FP8 task-quality suite before making any quality
      claim for FP8. Both baselines scored 9/13 across two fresh processes, with
      overlapping 95% Wilson intervals and no observed FP8 advantage. This
      bounded suite does not prove equivalence.

### Gate

- [x] Keep only correctness-qualified profiles active: retain the FP4 baseline,
      keep FP8 baseline as a quality-only control, and gate all ROCmFPX MTP
      profiles despite their performance frontier.
- [x] Promote no default from a single run. The repeated correctness evidence
      instead retains the existing FP4 baseline default.

## Stage 4: Extend the existing DwarfStar lane

**Value:** compare the specialized DeepSeek V4 runtime and its exact 0731
speculator without introducing another language-model download or host runtime.

The DS4 base path is already containerized and qualified on this gfx1151 host.
At 32K the hybrid control measured 97.73 PP tok/s, 12.56 decode tok/s, and
107.6 GiB peak GTT. Persistent disk KV reduced the repeated-prefix sample from
107.7 seconds cold to 3.7 seconds restored. Claims from Metal or CUDA systems
remain context only.

### Tasks

- [x] Pin the exact `DeepSeek-V4-Flash-DSpark-support-0731.gguf` support GGUF
      beside the installed Antirez hybrid. Do not reuse the Unsloth
      `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf`, which belongs to the standalone
      IQ3_XXS llama.cpp profile. The installed 5,989,114,272-byte companion
      matches SHA-256 `7e319924541db3f7a163ed7e11d7532a70d48228ab59d36cb81e1d4511885360`.
- [x] Add an opt-in `ds4-deepseek-v4-flash-hybrid-dspark-16k` profile using
      `--dspark-confidence 0.7`, context 16K, and `--prefill-chunk 1024` for its
      first memory-safe screen.
- [x] Add render/acquisition tests proving that the DS4 control selects only the
      main GGUF while the DSpark profile selects exactly the main plus its
      bounded support artifact.
- [x] Record the DS4 image digest and run a target-only control before the
      speculative candidate. Keep all execution and benchmarking in disposable
      or managed Podman containers. The resolved image digest is
      `sha256:2ea5b3b28334f08d53307baf79838591e510628d41dacec357de32ffafbac31f`;
      the fixed target-only sample completed in 105.025 seconds and its disk-KV
      restore control in 3.556 seconds.
- [x] Preserve the original greedy 16K diagnostic as failure evidence. The
      pre-fix image loaded the support model, but reported `proposed=0` and
      `accepted_draft=0` across 398 decode cycles.
- [x] Do not attempt 32K because the 16K proposal-activity prerequisite failed.
      Keep the existing target-only and disk-KV profiles as rollback controls.
- [x] Trace the failure to Antirez commit
      `84cc882352757baf628a1776badf7cc54d584e28`, “rocm: enable DSpark
      speculative decoding,” which landed after the original DS4 image.
- [x] Replace that image with a project-built `localhost/halo-ai-ds4:b0001`
      image. Pin the 164,211,638-byte `lemonade-sdk/ds4-rocm` b0001 archive,
      SHA-256 `e63b7c9428fd10de75b3f69164eccd564a8e7261c2f9087cd8bec2b4b19a8ad1`,
      exact source commit, bundled ROCm `7.15.0a20260728`, and immutable Ubuntu
      base digest. The final local manifest is
      `sha256:f9dd84e76c2fbdd3f99b2e0490c40b899586dd047b0bfa0030744cbd58e1df89`.
      Fail availability closed when any provenance label differs.
- [x] Re-run the deterministic 16K canary on gfx1151. The exact smoke response
      passed, the 400-token probe ran at 8.62 tok/s, and shutdown reported 324
      proposals with 246 accepted draft tokens (75.93%), zero verifier errors,
      and zero runtime errors.
- [x] Remove the obsolete local image after the fixed runtime passed. Retain
      target-only and disk-KV profiles as operational controls; speed is not a
      promotion requirement for this explicitly experimental DSpark profile.
- [x] Verify the installed hybrid GGUF itself advertises
      `deepseek4.context_length=1048576` with YaRN scaling from an original 65K
      context. Treat the model's architectural 1M limit as distinct from this
      host's safe runtime allocation.
- [x] Add persistent-KV DSpark work profiles at 128K and 256K while retaining
      16K as the minimal rollback. Both use the existing exact target/support
      pair, `--prefill-chunk 1024`, confidence 0.7, and the guarded 8 GiB disk
      KV policy; neither downloads another model.
- [x] Qualify both allocations on gfx1151. At 128K, post-probe GTT was 104.38
      GiB with 12.28 GiB `MemAvailable`; DS4 planned 1.78 GiB KV plus 0.25 GiB
      buffers. At 256K, GTT was 106.30 GiB with 10.36 GiB available; DS4 planned
      3.46 GiB KV plus 0.50 GiB buffers. Both returned exact smoke output and
      accepted the same 246/324 drafts with zero errors.
- [x] Add a separate 393,216-token `384k-think-max` profile because the pinned
      DS4 binary explicitly downgrades `reasoning_effort=max` below that exact
      threshold. Keep DSpark and persistent KV, plus add a request preset using
      DeepSeek's official `temperature=1.0`, `top_p=1.0` sampling defaults.
- [x] Qualify the 384K allocation and mode. Post-probe GTT was 108.23 GiB with
      8.40 GiB `MemAvailable`; DS4 planned 5.14 GiB KV plus 0.75 GiB buffers.
      A `reasoning_effort=max` request returned nonempty reasoning and answer
      content. Across both probes DSpark accepted 353/436 drafts (80.96%) with
      zero verifier or runtime errors.
- [x] Add first-class catalog profile aliases and map the `ds4` shortcut to the
      qualified 384K Think Max profile. Resolve aliases across lifecycle,
      inspection, smoke-test, and benchmark commands while retaining canonical
      profile IDs in runtime and trial state.

### Gate

- [x] **Pass after upstream fix:** enable the 16K profile because the fixed ROCm
      backend both proposed and accepted drafts with valid output. Keep it
      experimental because its 8.62 tok/s probe was slower than the target-only
      control; this lane optimizes for working DSpark behavior, not peak speed.
- [x] **Pass for practical long context:** enable 128K as the conservative work
      profile and 256K as the high-context profile. Both remain experimental
      until a long-prompt workload—not merely allocation plus a short probe—has
      exercised its intended frontier. Do not infer that 384K or 1M is safe from
      the model metadata alone.
- [x] **Pass for opt-in Think Max:** enable 384K because it retains twice the
      configured 4 GiB host reserve and exercised the max-effort response path.
      Prefer 256K for ordinary work; 384K is tighter and remains experimental.

## Deferred research

External NPU drafting, cross-version Qwen3.6 drafting, generalized multi-service
runtime plumbing, and vision/video qualification are outside the active PP/TPS
loop. Preserve their constraints and possible future work in
[`docs/TODO backup.md`](docs/TODO%20backup.md). Do not download another language
model or install host packages for these tracks without an explicit new gate.

## Active constraints and artifact policy

- The q38rocm FP4/FP8 GGUFs are text-only language/MTP artifacts; vision needs
  a separately qualified companion.
- Performance claims from other hosts are context only, never pass/fail evidence.
- Build 213 cannot compose `ngram-mod` with strict-Qwen MTP because their
  recurrent rollback requirements conflict.
- FP4 and FP8 are already present and verified. Selecting either profile must
  not download the other artifact.
- Do not download another language model. Container runtime updates and bounded
  support artifacts such as Ninja, templates, video fixtures, and a selected
  compatible mmproj remain allowed.
