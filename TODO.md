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
- A Qwen3.8 Flash Next ROCmFP4/Unsloth comparison is approved as the next
  download-backed experiment, but acquisition is intentionally deferred to the
  next work session. It must remain isolated from the qualified Qwen3.8 27B
  q38rocm runtime.

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
- [x] Add the FP4 artifact and a `qwen3.8-27b-rocmfp4-baseline` profile to the
      catalog with speculation and vision disabled.
- [x] Add profile-scoped acquisition with dry-run, resume, exact size/SHA
      verification, and atomic completion.
- [x] Make the acquisition dry-run prove that the baseline downloads no FP8,
      NPU model, visual projector, BF16 model, or reference quant.
- [x] Validate the GGUF header and tensor inventory. Trust the verified GGUF,
      not the repository's inconsistent companion `config.json`.
- [x] Add render, lifecycle, readiness, and deterministic smoke tests.
- [x] Change the Qwen3.8-27B default reasoning effort from the template's
      implicit `xhigh` to neutral `medium`. The live `/apply-template` canary
      proved omitted equals explicit medium and differs from xhigh; a GSM8K-
      style pilot and one 29,698-token pinned LongBench-v2 sample returned
      correct, nonempty final answers under default, medium, and xhigh.
- [ ] Estimate the reported intermittent empty-final-answer rate with at least
      24 repeated long-input agentic or structured-output calls per arm. Treat
      the current LongBench result as a bounded no-regression pass, not proof
      that medium is always faster, shorter, or more accurate than xhigh.
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

- [x] Add `qwen3.8-27b-rocmfp4-mtp` using the MTP tensors already stored in the
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
      `qwen3.8fp4` alias. Keep the two aggressive FP4 profiles and FP8 MTP gated
      pending a backend fix and complete identity-suite rerun.
- [x] **Operator-approved exception:** expose
      `qwen3.8-27b-rocmfp4-mtp-conservative-q5-draft` as an opt-in experimental
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
- [x] Add a no-download Lemonade ROCm/HIP A/B profile that reuses the exact Q6
      target and BF16 projector. Match the target-only Vulkan control at native
      262K context, F16 K/V, one slot, Flash Attention, and 4096/4096
      batch/ubatch, with speculation explicitly disabled. The vision canary
      passed on Lemonade 11.5.2 with ROCm package `b10597` / active binary
      `b10594`. A same-process three-run 81-prompt/256-generation comparison
      measured median decode at 8.05 tok/s HIP versus 8.44 Vulkan (Vulkan 4.9%
      faster), while the first uncached prefill measured 138.84 versus 65.10
      tok/s (HIP 2.13x faster). Keep this as a prompt-heavy target-only option;
      it does not replace DFlash2's roughly 24.5 tok/s qualified decode path.
- [x] Upgrade the managed Lemonade path to pinned 11.7.0 and test a first-class
      local target + BF16 projector + DFlash2 companion registration. Lemonade
      correctly emitted `--model-draft`, `--spec-type draft-dflash`, and
      `--spec-draft-n-max 5`, but stable package `b10597`/active build 10594 and
      nightly `b1315` both failed before inference with `expected 81, got 58`.
      The 58-tensor sidecar is Qwen3.8 DFlash2; Lemonade's binaries follow
      upstream DFlash v1 while DFlash2 remains open llama.cpp PR #27342. Keep
      `qwen3.8-27b-q6xl-vision-dflash2-lemonade` compatibility-gated and retest
      only after that PR or an equivalent loader lands. Do not run the
      `GPU_MAX_HW_QUEUES=1` arm for this error: tensor-schema validation occurs
      before HIP queue creation, so the variable cannot affect it. The opt-in
      knob remains available for a future backend-execution diagnostic.
- [x] Test the official Q8_0 and BF16 DFlash2 drafters as higher-fidelity
      controls after the combined vision path diverged at `n-max=7`. Across
      three fresh processes per arm, Q8_0 and BF16 produced the same alternate
      structured output, accepted 38/49 drafts, and averaged 23.24 and 22.76
      tok/s. Q4_K_M accepted 40/42 and averaged 27.54 tok/s at the same setting.
      This single canary does not establish a drafter-quality ordering: the
      later Q4_K_M `n-max` sweep proved verification-batch shape can select a
      different near-tie trajectory. Retain the larger profiles as diagnostics;
      keep the smaller, faster Q4_K_M behind `qwen3.8df2` at qualified `n-max=5`.
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

- [x] Keep `qwen3.8fp4` on the ROCmFP4 baseline. Map the operator-selected
      `qwen3.8df2` alias to the experimental Q6 vision+DFlash2 Q4_K_M profile at
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

## Stage 5: Compare Qwen3.8 Flash Next ROCmFP4, Unsloth, and DeepSeek V4 Flash

**Value:** determine whether the new 180B-class Flash Next ROCmFP4 layout is a
faster daily model on this Strix Halo without mistaking publisher results,
runtime improvements, or a different memory budget for a quantization win.
No Stage 5 model has been downloaded yet.

### Pinned research snapshot (2026-08-30 PDT)

- The new model is
  [`agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF`](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF/tree/ad4c5717254a630ee0c5a8db5208eb1f8476e56c),
  revision `ad4c5717254a630ee0c5a8db5208eb1f8476e56c`. The recommended split-PLE
  root file is `Qwen3.8-Flash-Next-ROCmFP4-FAST-v2-ple16.gguf`,
  `93,484,237,760` bytes (87.064 GiB), SHA-256
  `552a7a162f6a620c3aa0850d070086bc2b95094e0a4e8b860694c7f212cb59d8`.
  Prefer it over the byte-equivalent joined-table `v2/` layout because the
  latter requires an on-disk or host-RAM n-gram table.
- Its optional adaptive MTP companion is
  [`agentionai/Qwen3.8-Flash-Next-MTP-ROCmFP4-FAST-GGUF`](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-MTP-ROCmFP4-FAST-GGUF/tree/5a2cf56c3e0f8bc8d395d53bec649ec8e358a993),
  revision `5a2cf56c3e0f8bc8d395d53bec649ec8e358a993`. File
  `Qwen3.8-Flash-Next-MTP-ROCmFP4-FAST.gguf` is `2,444,519,296` bytes
  (2.277 GiB), SHA-256
  `5046d69571bb35c699e19c7a00b36633c2fdfdd06d70482551744ec523a6b590`.
  The optional F16 vision projector is `904,004,128` bytes (0.842 GiB),
  SHA-256 `f456cd796fdbdef0cadb22710b54e0071b6cdf22c07963365baa898267aec517`.
- The closest size-matched Unsloth control is
  [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/c8b5954a88c2775c546b92593eda40ea041d3176/UD-Q3_K_XL)
  `UD-Q3_K_XL`, revision
  `c8b5954a88c2775c546b92593eda40ea041d3176`: three shards totaling
  `89,986,353,824` bytes (83.806 GiB). Their SHA-256 values, in shard order,
  are `f2ef4328929d8b8c8930e2856eef52128dd4ce3425302f04bc3c657431cc4c49`,
  `7d230e7c9421d868b89eebaf23033af0ea1a4e046956df00fb156814fb62346e`,
  and `21d4f90f9cd7b7c3a1582667c20cb22f7b03de895b88a23bb20aaeaa44f2c199`.
  This is a fairer first control than `UD-Q4_K_XL` (103.69 GiB); a nominal
  Q-bit label is less useful than similar bytes resident on this APU.
- The installed DeepSeek V4 Flash Antirez hybrid remains the independent
  system control: `97,591,747,456` bytes (90.889 GiB), 97.73 prompt tok/s and
  12.56 decode tok/s in the existing direct 32K measurement. Its 384K DSpark
  Think Max lane remains the locally qualified capability/context control.
- Flash Next requires the
  [`LaurentZuijdwijk/llama.cpp` `vulkan/qwen4exp-rocmfpx` branch](https://github.com/LaurentZuijdwijk/llama.cpp/tree/vulkan/qwen4exp-rocmfpx),
  not the current q38rocm image. The observed branch head is
  `3466b48806f9fefe1162aa4053ffebcfcabe83aa`; re-check and pin the chosen
  commit immediately before building because this branch is moving quickly.

### q38rocm change review

- [`q38rocm` v1.5.3](https://github.com/julianmb/q38rocm/releases/tag/v1.5.3)
  finally maps its prebuilt binary to source
  `0fc9568e07ccc8553010864cb8db1957e629cbfa` (build 244), with SHA-256
  `10f060aa19ce9976f8807ecdacda8f708a13209cad0aa7c3111293ebe0ca5ad7`.
  It adds MTP prompt-cache checkpoint salvage and reduces router child-stop
  delay. Current main also documents 21x/44x/79x repeated-prefix cache reuse at
  32K/64K/130K and improves release automation, model paths, and defaults.
- Those are useful 27B operational changes, but they do not provide the
  qwen4exp/ROCmFPx per-head-PLE runtime required by Flash Next. Conversely, the
  Flash Next fork is not evidence that the existing 27B engine should change.
- Halo already rejected build 244 at `0fc9568` as a drop-in update because it
  produced nonsensical output with the installed 27B ROCmFP4 GGUF. Preserve
  the qualified v1.0/build-213 image. Any v1.5.3/cache experiment must use a
  separate image and repeat the exact smoke and 13-case quality canaries first.

### Working hypothesis, not qualification evidence

- The publisher reports 245.5 prompt tok/s and 24.67 no-MTP decode tok/s at
  32K. Against this host's existing DeepSeek control that is a tempting 2.51x
  prefill and 1.96x decode hypothesis, but it is not an A/B result: the host,
  engine, prompt, quant, and model all differ. Reported MTP reaches up to 40
  tok/s and is explicitly content-dependent.
- ROCmFP4 plus its MTP file occupies about 89.34 GiB before KV and runtime
  allocations, versus 83.81 GiB for Unsloth UD-Q3_K_XL and 90.89 GiB for the
  installed DeepSeek hybrid. All should be feasible candidates, but only a
  monitored allocation test can establish the safe context ceiling.
- Qwen offers a 262K context target and optional vision. DeepSeek retains the
  larger architectural context, locally qualified 384K Think Max path, and a
  different reasoning/coding capability profile. “Faster daily model” and
  “best difficult-task model” are separate decisions.

### Tomorrow-session tasks

- [ ] Re-read all three repository heads before acquisition. Keep the revisions,
      filenames, byte counts, and SHA-256 values above unless an upstream
      change is deliberately reviewed and re-pinned; never download from
      mutable `main` or `latest`.
- [ ] Add resumable, cache-reusing, atomic acquisition for the split-PLE
      ROCmFP4 target and size-matched Unsloth shards. Keep MTP and the projector
      optional so a baseline selection does not fetch them. Prove a second
      acquisition reuses the verified local files without network transfer.
- [ ] Build the Laurent branch as a new digest-pinned candidate engine. Record
      source commit, build flags, base image digest, `llama-server --version`,
      Mesa/Vulkan identity, and ROCm packages. Do not replace or retag the
      current q38rocm image.
- [ ] First load each target at 32K with Q8 K/V, Flash Attention, no MTP, and
      no prompt cache. Verify qwen4exp metadata, the intended PLE placement,
      exact smoke output, chat/tool template behavior, peak GTT, host
      `MemAvailable`, swap activity, kernel errors, and absence of a GPU reset.
- [ ] Run Unsloth UD-Q3_K_XL on stock mainline and on the candidate fork where
      supported. Run ROCmFP4 on the candidate fork. This separates runtime
      benefit from quant/layout benefit; do not call a fork-vs-stock result a
      ROCmFP4-vs-Unsloth result.
- [ ] Add the ROCmFP4 MTP companion only after target-only passes. Compare
      target-only with adaptive draft lengths 2-4 and record proposed/accepted
      tokens by workload. Keep exact greedy output identity and sampled
      recommended-settings quality as separate trials.
- [ ] Run the fixed 13-case suite twice in fresh processes for both Qwen quants,
      followed by long-context retrieval, structured tool calls, vision when
      the projector is selected, and two or three real repository coding tasks.
      Stable and equal quality is a pass; do not require a synthetic score win.
- [ ] Benchmark cold, unique prompts at 4K, 16K, 32K, and 128K with fixed
      completion lengths and at least three interleaved repetitions. Re-run the
      direct DeepSeek 32K control in the same session. Record PP, TTFT, decode,
      end-to-end time, GTT/VRAM, `MemAvailable`, PPT/TDP, and MTP acceptance.
      Make same-session throughput and energy ratios primary because this
      machine's power limit may differ from earlier measurements.
- [ ] Run warm repeated-prefix/cache tests as a separate table only after the
      cold matrix. Label cache reuse explicitly; never mix q38rocm's reported
      21x/44x/79x reuse with uncached model throughput.
- [ ] Expand context only after 32K passes: 128K first, then 262K if at least
      8 GiB host `MemAvailable` remains after the probe and there are no GPU or
      allocation errors. DeepSeek's existing 384K profile is the rollback and
      context-capability control, not proof that Qwen can allocate the same.

### Gate

- [ ] Promote the new ROCmFP4 profile only if it is repeatably sane, stable or
      equal in the fixed and practical quality checks versus the selected
      Unsloth control, and has a meaningful same-session speed/energy or memory
      advantage. A publisher benchmark alone cannot pass the gate.
- [ ] Keep DeepSeek V4 Flash available regardless of the Qwen outcome. Qwen may
      become the interactive default while DeepSeek remains the stronger
      long-context/Think Max option; let measured work decide rather than force
      one global winner.
- [ ] Do not promote q38rocm v1.5.3 for the existing 27B model unless its
      isolated candidate reruns eliminate the previously observed build-244
      output corruption and pass the full regression suite.

## Stage 6: Qualify the best-current speech runtime

**Value:** test the strongest currently published Python, ROCm, PyTorch, and
application stack first, then promote it only if the complete Seamless M4T v2
service wins a same-host comparison with the qualified image. Do not spend the
first trial stepping through intermediate versions merely to reduce the number
of changed components; preserve those versions as bisection points only if the
preferred candidate fails.

### Preferred challenger

- Base image: `python:3.14.7-slim-trixie`.
- GPU stack: ROCm `10.0.0`, PyTorch `2.13.0`, torchvision `0.28.0`, and
  torchaudio `2.11.0.2`, using AMD's CPython 3.14 `gfx1151` wheels from
  `https://stable.repo.amd.com/rocm/whl-next/`.
- Model/application stack: Transformers `5.16.1`, safetensors `0.8.0`,
  tiktoken `0.14.0`, Accelerate `1.14.0`, SoundFile `0.14.0`, SentencePiece
  `0.2.2`, protobuf `7.36.0`, SciPy `1.18.1`, Gradio `6.26.0`, FastAPI
  `0.141.1`, Uvicorn `0.52.4`, and python-multipart `0.0.32`.
- Control: retain the qualified Python 3.12 / ROCm 7.14 / PyTorch 2.12 /
  Transformers 4.57.1 / Gradio 6.16 image unchanged, including its immutable
  identity and current `SPEECH_IMAGE` selection.
- Candidate identity: build and record a separate immutable image such as
  `localhost/halo-ai-speech:rocm-10.0-py3.14`; never reuse or retag the control
  while qualification is incomplete.

The CPython 3.14 Linux wheel-only application graph resolved successfully in
the 2026-08-30 research pass, and AMD publishes the matching ROCm 10/PyTorch
device wheels. That proves availability, not runtime correctness. The target
CachyOS host is outside AMD's formally listed ROCm 10 `gfx1151` Linux host
matrix, so only a live same-host trial can pass this stage.

### Qualification tasks

- [ ] Re-check every upstream package and image immediately before the build.
      Pin the base-image digest, exact wheel versions, resolved transitive
      dependencies, source indexes, and resulting image manifest. Generate a
      machine-readable lock or build manifest and fail closed on an unexpected
      source or version.
- [ ] Add the preferred challenger as an isolated build path and tag. Keep the
      current Containerfile/image available as the control, and do not update
      CLI, installer, or example-config defaults during the experiment.
- [ ] Make the candidate build run `pip check` and import the complete service
      dependency set. Record Python, Torch, HIP, torchvision, torchaudio,
      Transformers, Gradio, NumPy, SciPy, and ROCm device-package versions in
      image labels or a retained manifest.
- [ ] On the `gfx1151` host, verify `torch.cuda.is_available()`, device name,
      `torch.version.hip == "10.0.0"`, architecture coverage, a real GPU tensor
      operation, and clean ROCm initialization before loading model weights.
      Treat detection without successful compute as a failure.
- [ ] Start the challenger with the existing hash-pinned
      `facebook/seamless-m4t-v2-large` model mounted read-only. Require offline
      processor/model loading, the same 36 speech-input/output languages, and a
      ready health response that exposes exact Python/Torch/HIP/application
      versions.
- [ ] Run the existing AMD-sample Spanish translated-WAV smoke test against
      both control and challenger. Add repeated fresh-process and warm-process
      trials, plus at least one additional language, and inspect duration,
      sample rate, finite samples, clipping, silence, and intelligibility. A
      nonempty WAV alone is necessary but not sufficient evidence of parity.
- [ ] Benchmark the two images in an interleaved same-session comparison.
      Record image size, build time separately from runtime, cold model-load
      time, first-request latency, warm inference latency, output duration,
      peak GTT/VRAM, host `MemAvailable`, CPU/RSS, and run-to-run variability.
      Use identical model bytes, source audio, target languages, power profile,
      and completion conditions.
- [ ] Run a bounded stability soak with repeated translations while monitoring
      container logs and the host kernel journal for GPU faults, resets, queue
      stalls, allocation failures, or ROCm warnings. Record kernel, firmware,
      Mesa/amdgpu, Podman, and host-profile identities with the result.
- [ ] If the preferred challenger fails, bisect instead of guessing. First
      retain Python 3.14/ROCm 10/Torch 2.13 while reverting the Transformers 5
      and Gradio 6.26 application pair; next retain Python 3.14 while reverting
      to ROCm 7.14/Torch 2.12 and CPython-3.14-compatible SciPy/tiktoken; use
      Python 3.13 with the original application pins only as the final narrow
      compatibility control. Stop at the highest stack that passes the same
      gate.
- [ ] After a pass, remove only dependencies proven unnecessary by a second
      isolated image and full rerun. `torchvision`, SciPy, tiktoken, and
      Accelerate are not direct imports in `speech_server.py`, but static
      inspection is not proof that Transformers processor/model paths never
      require them.
- [ ] Promote only after qualification: update the canonical Containerfile,
      `SPEECH_IMAGE` defaults, installer text, example configuration,
      documentation, expected version assertions, and recorded image identity
      together. Preserve the prior immutable image tag and document the exact
      rollback command.

### Gate

- [ ] **Preferred-candidate pass:** build and dependency checks succeed; real
      `gfx1151` compute, offline model load, API/UI health, language discovery,
      translation quality, and the stability soak pass without GPU or kernel
      errors. Performance and memory must be at least within measured control
      variability, or a documented lifecycle/security benefit must justify a
      small bounded regression.
- [ ] **Win and promote:** prefer the ROCm 10/Python 3.14 challenger when it is
      functionally equivalent and either materially improves latency, memory,
      stability, or image maintenance, or remains performance-neutral while
      moving the runtime to the current published stack. Report all metrics,
      including losses; “newer” by itself is not a benchmark win.
- [ ] **Fail and retain control:** keep the current ROCm 7.14 image and defaults
      when the challenger corrupts output, loses required languages, exhibits
      unresolved host instability, or regresses materially. Record the first
      passing bisection candidate, but do not promote it without the complete
      control comparison.

## Deferred research

External NPU drafting, cross-version Qwen3.6 drafting, generalized multi-service
runtime plumbing, and vision/video qualification are outside the active PP/TPS
loop. Preserve their constraints and possible future work in
[`docs/TODO backup.md`](docs/TODO%20backup.md). Do not download another language
model or install host packages for these tracks without an explicit new gate.
Stages 3B and 5 are the approved exceptions. Stage 3B permits only its one Q6
target plus the exact DFlash2 and vision support artifacts listed there; Stage
5 permits only the pinned Flash Next ROCmFP4, optional MTP/projector, and
size-matched Unsloth artifacts above, beginning in the next work session.

## Active constraints and artifact policy

- The existing q38rocm 27B FP4/FP8 GGUFs are text-only language/MTP artifacts.
  Flash Next vision needs its separately acquired and qualified projector.
- Performance claims from other hosts are context only, never pass/fail evidence.
- Build 213 cannot compose `ngram-mod` with strict-Qwen MTP because their
  recurrent rollback requirements conflict.
- FP4 and FP8 are already present and verified. Selecting either profile must
  not download the other artifact.
- Do not download another language model beyond the exact Stage 3B and Stage 5
  targets. Their listed sidecars are approved bounded companions; unlisted
  quant tiers and redundant projectors remain excluded.
