# Qwen3.8 ROCmFPX roadmap

The goal is to add Qwen3.8 support to Halo AI in the order that produces the
most value for the least downloading and engineering work.

## Decisions

- The first and only required baseline is the text-only
  `Qwen3.8-27B-ROCmFP4-FAST.gguf` running on the iGPU without NPU drafting.
- Built-in GPU MTP was the first acceleration tried because it uses the same
  model file and requires no draft-model download. The conservative FP4 policy
  is available only as an explicit experimental profile: it retained the
  baseline's 9/13 bounded quality score, but strict output identity is not
  proven. The two aggressive FP4 policies and FP8 MTP remain gated.
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
  conservative MTP matched 11/13. A 2026-08-23 cache-isolation follow-up
  (`cache_ram_mb=0`, both no-cache flags, and slot similarity zero) matched
  13/13 only when no request preceded the suite. Under Halo's normal
  smoke-then-suite lifecycle it still differed on code slicing (12/13), so it
  is not production-history invariant. Aggressive no-cache MTP matched 8/13.
  The strict-Qwen backend flag therefore does not establish end-to-end greedy
  identity on this build.
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

- [x] **Strict gate failed:** retain the unassisted FP4 baseline and
      `qwen38fp4` alias. Keep the two aggressive FP4 profiles and FP8 MTP gated
      pending a backend fix and complete identity-suite rerun.
- [x] **Operator-approved exception:** expose
      `qwen38-27b-rocmfp4-mtp-conservative-q5-draft` as an opt-in experimental
      profile because its bounded quality score remained 9/13 and its observed
      divergence was accepted; do not describe it as lossless or make it the
      default alias.

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
but loses 10.41% and 24.16% decode throughput. The subsequent strict-identity
gate disqualified every MTP policy from default promotion. The conservative FP4
profile remains available as an explicit experimental exception; FP8 uses
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

- [x] Keep default and alias choices correctness-qualified: retain the FP4
      baseline, keep FP8 baseline as a quality-only control, expose conservative
      FP4 MTP only as an explicitly accepted experimental exception, and gate
      the other three ROCmFPX MTP profiles.
- [x] Promote no default from a single run. The repeated correctness evidence
      instead retains the existing FP4 baseline default.

## Stage 3B: Qualify one Vulkan DFlash2 alternative

**Value:** test the strongest published same-device speed/quality candidate
without downloading an entire quant ladder.

### Research decision

- [x] Select exactly one target quant:
      `unsloth/Qwen3.8-27B-GGUF` `UD-Q6_K_XL`. Initial research selected Q5,
      but direct inspection of Unsloth's more relevant Divergence-300@32 graph
      changed the decision: Q5 XL is near 70% 32-token trajectory agreement,
      while the Q6 tier is roughly 77--79%. The approximately 7--9 point
      absolute gain is large enough to matter for agentic/tool trajectories.
      Q6 costs 21% more weight bytes, and a separate same-device MTP comparison
      measured it about 15% slower than Q5 target-only; this is the preferred
      quality/speed trade on a 128 GB Strix Halo.
- [x] Verify the exact release generation rather than relying on the repository
      title. The selected current Q6 file was uploaded in the 2026-08-19
      Dynamic 3 release and differs from the 2026-08-14 preview
      (25,924,152,384 bytes, SHA-256 `739202186fd9389bb58497c58b56c8a0d4253d99d20131e6a0427e363e678fc8`).
      However, Unsloth explicitly says its gains were small on larger quants and
      it retained the older Dynamic 2 methodology there. Describe Q6 as the
      current Dynamic 3 release artifact with a retained large-quant recipe,
      not as an all-new Dynamic 3 layer layout.
- [x] Reject Q5 and Q8 *target-model* downloads for the qualified trial. The reported Q5
      DFlash2 result—31.4 tok/s at 80 W and 30.2 tok/s at 70 W—remains a useful
      speed hypothesis, but it does not outweigh the observed Q6 trajectory
      gap. Q8 is larger again, while fidelity measurements above roughly 25 GB
      converge toward noise. No published same-device DFlash2 run establishes
      Q6 throughput, so Halo must measure it rather than extrapolate.
- [x] Include only the matching 1,143,006,752-byte DFlash2 Q4_K_M draft GGUF
      required by the selected acceleration path. Treat it as a bounded support
      artifact, not a second target-model choice.
- [x] Include the target repository's 931,146,432-byte BF16 vision projector,
      but not its redundant F16 projector. The main GGUF already embeds runtime
      tokenizer/model/template metadata; README and `config.json` are not
      runtime dependencies.

### Exact artifact pins

- Target: `unsloth/Qwen3.8-27B-GGUF` revision
  `4ca720788d1e01f1bff70c033e0d0028fd02e502`, file
  `Qwen3.8-27B-UD-Q6_K_XL.gguf`, 25,299,061,664 bytes, SHA-256
  `701d8fa9ed214ab21bfc130cd2a7df19ca89bbef7713e2dfb19f3c63696aa917`.
- Vision: same revision, `mmproj-BF16.gguf`, 931,146,432 bytes, SHA-256
  `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53`.
- Drafter: `incoai/Qwen3.8-27B-DFlash2-GGUF` revision
  `6cb5872e2cee6b4e780a8414922350be8e42d65c`, file
  `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, 1,143,006,752 bytes, SHA-256
  `18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594`.
- Diagnostic drafters from the same revision: `Qwen3.8-27B-DFlash2-Q8_0.gguf`,
  2,056,414,752 bytes, SHA-256
  `7f1c9a31a6ed40044c69f6508b50fd63b87abd8e1fb7fe4290303df549153751`;
  and `Qwen3.8-27B-DFlash2-BF16.gguf`, 3,860,293,152 bytes, SHA-256
  `26af33a15b21475d668e4ee55639beea49932e7360b1144c6282721bcd127c14`.
- Text reproduction control: Nathan's `dev-20260819-0b0f35d` image at
  `sha256:041e6491a241e48a75bd900644783eb20b2d0da5104f4c4d55f7ed1932b8b4a8`.
  DFlash2 support is still carried outside upstream llama.cpp's merged release
  line, so this runtime is experimental.
- Vision candidate: `dev-20260822-f25eefe` source
  `f25eefeaf0386c18499f23f4fc4f400397638d51`, image
  `sha256:a4a3dfe5813df1f0687e526bcba639bfd8b7cdb1d5e4c1c855abc240fb574d3a`.
  It contains the later M-RoPE DFlash draft-cache alignment fix; do not enable
  vision speculation merely because the artifacts load.

### Qualification tasks

- [x] Acquire and fully verify only the selected Q6 target, BF16 projector, and
      DFlash2 Q4_K_M sidecar. Their exact byte counts, GGUF headers, and SHA-256
      digests match the catalog; no Q5 or Q8 target remains on disk.
- [x] Add a distinct engine identity for Nathan's prerelease Strix Halo Vulkan
      fork; do not replace the ordinary llama.cpp image globally.
- [x] Add target-only, text+DFlash2, and target-only vision profiles at the
      model's native 262,144-token context. Keep
      DFlash2 off for vision until the M-RoPE fixed runtime passes paired image
      correctness tests. The target-only vision profile passed the red-image
      canary with the pinned BF16 projector. A controlled vision+DFlash2 trial
      on build 10577 proved that drafting executes (40/42 proposals accepted
      on a 46-token structured response) and raised decode from 8.44 to 28.00
      tok/s at `n-max=7`, but it did not match the target-only greedy output.
      Reducing the verification block to `n-max=5` matched the target-only
      structured response in all three fresh processes and still averaged
      24.53 tok/s versus 8.43 target-only. The combined profile therefore uses
      5; target-only vision remains the rollback.
- [x] Prove target-only versus DFlash2 greedy token identity across fresh
      processes and after request-history perturbation. The drafter's
      distribution-preserving design claim does not substitute for this test.
      The first fresh-process matrix matched 12/13 output-token sequences; the
      code-trace case diverged (`-1` baseline, `1` DFlash2), so this gate has
      not passed at `n-max=7` even though both profiles scored 10/13. A controlled retry
      with both engines freshly started and no preceding smoke request matched
      exactly (`-1`, tokens `[12,16]`) and accepted one of seven DFlash drafts.
      The completed equal-history rerun at `n-max=5` used the same vision smoke
      request before every suite and matched 13/13 output-token streams in both
      fresh-process repetitions. Both arms were independently stable 13/13 and
      scored 10/13; DFlash accepted 57/110 proposals in each repetition.
- [x] Run the fixed task-quality suite and same-host 4K/32K performance matrix.
      Retain Q6 only if its quality is competitive with the current FP4
      baseline and its end-to-end latency materially improves.
      Q6 scored 10/13 versus the recorded FP4 baseline's 9/13. At 4K+64,
      DFlash2 improved decode from 8.27 to 15.46 tok/s and reduced wall time
      6.9%; at 32K+64 it improved decode from 7.77 to 14.14 tok/s but increased
      wall time 8.7% because prefill fell from 175.64 to 158.03 tok/s. Peak GTT
      was 72.91--73.06 GiB versus 60.80--60.84 GiB target-only.
- [x] Test the official Q8_0 and BF16 DFlash2 drafters as higher-fidelity
      controls after the combined vision path diverged at `n-max=7`. Across
      three fresh processes per arm, Q8_0 and BF16 produced the same alternate
      structured output, accepted 38/49 drafts, and averaged 23.24 and 22.76
      tok/s. Q4_K_M accepted 40/42 and averaged 27.54 tok/s at the same setting.
      This single canary does not establish a drafter-quality ordering: the
      later Q4_K_M `n-max` sweep proved verification-batch shape can select a
      different near-tie trajectory. Retain the larger profiles as diagnostics;
      keep the smaller, faster Q4_K_M behind `qwen38df2` at qualified `n-max=5`.
- [ ] **Deferred; do not pursue yet:** maintain a local experimental branch of
      Nathan's `strix-halo-vulkan` llama.cpp fork only if configuration-level
      qualification still shows consequential target divergence. Start from
      source commit `f25eefea`, retain its M-RoPE draft-cache fix `64e2b680`,
      and build a separately tagged image so the pinned runtime remains an
      untouched rollback. First add target-logit/top-two-margin instrumentation;
      only then consider an alternate vision-position patch or guarded
      sequential re-verification near ties. Trigger this work only after a
      broader equal-history or real-image suites fail consequentially at the
      selected smaller `n-max`; do not
      create or publish forks merely for the synthetic red-image canary.
      When revisiting, re-audit the then-current llama.cpp/DFlash2 options and
      expose diagnostic profiles for any still-relevant knobs that Halo does
      not currently model: `--spec-draft-p-min`,
      `--[no-]spec-draft-backend-sampling`, draft K/V precision through
      `--spec-draft-type-k` / `--spec-draft-type-v`, and, if relevant to the
      updated implementation, `--spec-draft-n-min` or
      `--spec-draft-p-split`. Keep target `--backend-sampling` disabled during
      exact-output comparisons. Record proposal acceptance and speed separately
      from target parity: these controls can change the proposed tokens and
      verification batch shape, but none is an exactness guarantee. Also check
      whether upstream has restored configurable DFlash draft sampling, which
      was removed during the current PR's review.

### Gate

- [x] Keep `qwen38fp4` on the ROCmFP4 baseline. Map the operator-selected
      `qwen38df2` alias to the experimental Q6 vision+DFlash2 Q4_K_M profile at
      `n-max=5`. Its paired vision canary matched target-only in three fresh
      processes and its equal-history suite matched 13/13 token streams in two
      fresh-process repetitions. Preserve both target-only controls because the
      speed advantage still depends on the input/output ratio and the backend
      remains a prerelease fork.

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
Stage 3B is the sole approved exception and permits only its one Q6 target plus
the exact DFlash2 and vision support artifacts listed there.

## Active constraints and artifact policy

- The q38rocm FP4/FP8 GGUFs are text-only language/MTP artifacts; vision needs
  a separately qualified companion.
- Performance claims from other hosts are context only, never pass/fail evidence.
- Build 213 cannot compose `ngram-mod` with strict-Qwen MTP because their
  recurrent rollback requirements conflict.
- FP4 and FP8 are already present and verified. Selecting either profile must
  not download the other artifact.
- Do not download another language model beyond the single Stage 3B Q6 target.
  Its exact Q4_K_M, Q8_0, and BF16 DFlash2 diagnostic sidecars and BF16
  projector are approved bounded companions; Q5, Q8, alternate Qwen3.8
  *targets*, and the redundant F16 projector remain excluded.
