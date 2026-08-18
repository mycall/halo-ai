# Qwen3.8 ROCmFPX roadmap

The goal is to add Qwen3.8 support to Halo AI in the order that produces the
most value for the least downloading and engineering work.

## Decisions

- The first and only required baseline is the text-only
  `Qwen3.8-27B-ROCmFP4-FAST.gguf` running on the iGPU without NPU drafting.
- Built-in GPU MTP is the first acceleration to try because it uses the same
  model file and requires no draft-model download.
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

## Stage 3: Optimize same-host long-context PP/TPS

**Value:** improve the working Qwen3.8 GPU profiles before adding another
runtime architecture or model family.

### Current measured state

The aligned cold-cache benchmark uses exact 4,095- and 31,998-token arrays,
64 generated tokens, container-local requests, and 50 ms memory sampling.
These first-pass rows are single measurements; repeat finalists before changing
defaults.

| Profile | 4K PP / TPS | 32K PP / TPS | Peak GTT at 32K |
| --- | ---: | ---: | ---: |
| FP4 baseline | 187.44 / 11.75 | 122.61 / 10.70 | 14.02 GiB |
| FP4 MTP | 174.04 / 22.30 | 108.31 / 17.44 | 19.25 GiB |
| FP8 baseline | 188.90 / 7.60 | 119.58 / 6.87 | 26.00 GiB |
| FP8 MTP | 174.41 / 22.50 | 106.11 / 15.85 | 31.31 GiB |

FP4 MTP becomes faster end-to-end than FP4 baseline after about 42 generated
tokens at 4K and 954 tokens at 32K on this synthetic repeated-context workload.
FP8 is effectively tied with FP4 MTP at 4K, loses 9.12% decode throughput at
32K, and uses about 12 GiB more GTT. Keep FP8 as a quality experiment, not a
performance default.

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
- [ ] Repeat the FP4 baseline and FP4 MTP finalists at least three times using
      the non-repeating prompt pattern; report median and variability.
- [ ] Audit and tune only locally supported batch, micro-batch, thread-batch,
      KV-cache, and execution options, one axis at a time.
- [ ] Tune FP4 MTP proposal settings after the best underlying PP configuration
      is selected; retain acceptance and end-to-end crossover evidence.
- [ ] Run a fixed FP4-versus-FP8 task-quality suite before making any quality
      claim for FP8.

### Gate

- [ ] Keep only profiles on the measured PP/TPS/memory Pareto frontier.
- [ ] Promote no default from a single run; require stable repeated evidence and
      a clean lifecycle with no OOM or device reset.

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
