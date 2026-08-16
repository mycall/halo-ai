# Qwen3.8 ROCmFPX, XDNA2 NPU Drafting, and Vision

This document tracks the integration of the
[`q38rocm`](https://github.com/julianmb/q38rocm) Qwen3.8 27B ROCmFP4 runtime
into Halo AI, an optional GPU-only ROCmFP8 quality track, the experimental use
of the Strix Halo XDNA2 NPU as a speculative draft-model accelerator for the
FP4 target only, and a separately gated path to vision-language support.

The work is deliberately gated. The first and only required baseline is
`Qwen3.8-27B-ROCmFP4-FAST.gguf` on the GPU with no NPU, vision companion, or
additional model download. It should ship independently if every optional
track is deferred or rejected.

## Target architecture

```text
Client (text, and optionally image/video)
  | OpenAI API
  v
ROCmFPX verifier
Qwen3.8 27B ROCmFP4-FAST, Vulkan / iGPU
  |
  +---- optional built-in MTP, same GPU and model artifact
  |
  +---- later token-level socket ---- FastFlowLM NPU drafter
  |                                    tiny Qwen3.8-family Q4NX
  |
  `---- authoritative verification, sampling, and response
```

The verifier remains authoritative. The NPU proposes tokens only, so a bad
proposal cannot change correctness.

The drafter must be a small model from the same exact Qwen3.8 family, not
merely an unrelated model with a compatible tokenizer. Only the FP4 target is
in scope for external NPU drafting. ROCmFP8 remains an optional GPU-only
quality experiment; do not create an FP8+NPU profile or attempt to place the
27B FP8 target on the NPU.

The official
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) base model is a
native image/video language model. The available q38rocm GGUFs contain its
language decoder and MTP tensors, but not the vision encoder. Vision support
therefore requires a compatible visual projector/encoder artifact and backend
support; it does not require choosing FP8 over FP4.

| Target | Draft provider | Intended status |
| --- | --- | --- |
| ROCmFP4 | None | Supported baseline |
| ROCmFP4 | Built-in MTP on GPU | Supported performance profile |
| ROCmFP4 | Compatible model on NPU | Experimental until qualified |
| ROCmFP8 (`Q8_0_ROCMFPX`) | None | Optional GPU-only quality experiment; explicit download |
| ROCmFP8 (`Q8_0_ROCMFPX`) | Built-in MTP on GPU | Optional GPU-only quality/performance experiment |

The current q38rocm
[`npu_sidecar_drafter.py`](https://github.com/julianmb/q38rocm/blob/ae3d4640041c92f4b0a012d6bed004e0ad9e66b9/scripts/npu_sidecar_drafter.py)
does not run inference on the NPU. It detects the NPU, prints estimates, and
launches the same GPU MTP mode as the normal server. Genuine NPU drafting
therefore requires a new draft provider.

## Definition of done

The **initial FP4 baseline** is complete when:

- [ ] Halo acquires and verifies only
      `Qwen3.8-27B-ROCmFP4-FAST.gguf` and the pinned ROCmFPX runtime.
- [ ] The unassisted text-only profile starts, passes health and deterministic
      smoke tests, and stops cleanly through the normal Halo lifecycle.
- [ ] Its output and performance reproduce the pinned direct q38rocm invocation
      within an agreed tolerance.
- [ ] Acquisition records confirm zero FP8, NPU-model, projector, BF16, or
      reference-quant bytes were downloaded.

NPU drafting is **functional** when:

- [ ] The NPU genuinely proposes token IDs and XDNA execution is confirmed by
      device telemetry.
- [ ] The drafter has verified Qwen3.8 family lineage, tokenizer identity, and
      prompt/thinking semantics; generic Qwen3 or unrelated models are
      rejected.
- [ ] The GPU target verifies every proposal and remains the only authority for
      emitted tokens.
- [ ] Strict greedy output is token-for-token identical to unassisted target
      decoding on the conformance corpus.
- [ ] The runtime never silently falls back to CPU or GPU drafting.

NPU drafting is a **successful acceleration** when:

- [ ] It satisfies all functional requirements.
- [ ] It has no firmware resets, driver faults, hangs, or resource leaks during
      the qualification soak.
- [ ] It improves median end-to-end generation throughput by at least 10% over
      the FP4 GPU MTP profile on the representative benchmark corpus.
- [ ] Its p95 latency and power behavior remain within the agreed release
      limits.

If the acceleration requirements fail, ship the unassisted FP4 baseline and
its independently qualified GPU-MTP profile while keeping NPU drafting
disabled and explicitly experimental.

## Phase 0: Architecture, licensing, and provenance

### Design record

- [ ] Add an architecture decision record covering:
  - A dedicated `rocmfpx` verifier engine.
  - A FastFlowLM-based, same-family Qwen3.8 NPU drafter.
  - The smallest NPU-native quantization that provides the best measured
    latency/acceptance tradeoff and leaves operational memory headroom.
  - A versioned Unix-domain-socket protocol.
  - Strict greedy speculation as the first supported mode.
  - Unassisted FP4 as the supported default and GPU MTP as an optional
    performance profile.
  - Stop/go criteria for each experimental stage.
- [ ] Document why `rocmfpx` must not masquerade as ordinary `llamacpp`:
      ROCmFP4, TurboQuant, and the required runtime patches are a distinct
      engine ABI.

### Immutable inputs

- [ ] Pin the q38rocm repository, ROCmFPX fork, FP4 model, and runtime image by
      immutable commit, revision, or digest for the baseline.
- [ ] Record FastFlowLM, XRT, amdxdna, draft-model, NPU-kernel, FP8, and vision
      pins as optional-track metadata; resolve and acquire them only when that
      track reaches its own start gate.
- [ ] Pin the authoritative base model separately:
  - Repository: `Qwen/Qwen3.8-27B`
  - Initial audited revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
  - Architecture: `Qwen3_5ForConditionalGeneration`
  - Text vocabulary: 248,320 tokens
  - Modalities: text, image, and video
- [ ] Add the ROCmFP4 target artifact to the catalog with:
  - Repository: `julianmb/Qwen-3.8-27B-ROCmFP4-FAST-GGUF`
  - Revision: `e439c125804f73ccb1d9e6aebb5ea50f112833fa`
  - File: `Qwen3.8-27B-ROCmFP4-FAST.gguf`
  - Size: `14562236384` bytes
  - SHA-256: `fb89c78d2be91cdb68eaaaa45b1270710bf34aa721dc1f0b9e3aa7b98d2e1da9`
  - GGUF architecture: `qwen35`
  - GGUF file type: `103`
  - Tensor count: `866`
- [ ] Record the ROCmFP8 target as disabled, on-demand catalog metadata only;
      it must not be included in baseline acquisition:
  - Repository: `julianmb/Qwen-3.8-27B-ROCmFP4-FAST-GGUF`
  - Revision: `e439c125804f73ccb1d9e6aebb5ea50f112833fa`
  - File: `Qwen3.8-27B-ROCmFP8.gguf`
  - Quantization: `Q8_0_ROCMFPX`
  - Size: `28193396704` bytes
  - SHA-256: `0bf5bfc9f946090af2d41b388ccb4d627e916c7250517c36a0de37d6eaccfd8e`
  - GGUF architecture: `qwen35`
  - GGUF file type: `111`
  - Tensor count: `866`
- [ ] Store quantization, expected memory class, and supported modalities as
      catalog metadata rather than inferring them from filenames.
- [ ] Add a GGUF metadata inspection step that validates the actual file header
      and tensor inventory rather than trusting companion `config.json` files.
- [ ] Record the current provenance discrepancy:
  - The q38rocm companion config describes a different, text-only
    `Qwen3ForCausalLM` with a 152,064-token vocabulary.
  - Both audited GGUF headers instead identify `qwen35`, 65 blocks, a
    248,320-token vocabulary, the official Qwen3.8 dimensions, and one MTP
    layer.
  - Each GGUF has 866 language/MTP tensors and no vision, visual, image, video,
    patch, or merger tensors.
  - The q38rocm model repository contains no separate visual projector.
- [ ] Trace the GGUF conversion back to the pinned official base weights and
      verify tensor names, shapes, tokenizer, chat template, and MTP layout.
- [ ] Treat the q38rocm companion config as untrusted until it is corrected;
      never derive tokenizer or architecture compatibility from it.
- [ ] Reconcile the q38rocm build scripts' inconsistent source revisions and
      Vulkan/ROCm build options.
- [ ] Remove permissive source checkout behavior such as `git checkout ... ||
      true`; builds must fail on a missing or mismatched revision.
- [ ] Produce a reproducible container build with image labels, dependency
      versions, build manifest, and SBOM.

### Hugging Face repository audit

The pinned
[`julianmb/Qwen-3.8-27B-ROCmFP4-FAST-GGUF`](https://huggingface.co/julianmb/Qwen-3.8-27B-ROCmFP4-FAST-GGUF/tree/e439c125804f73ccb1d9e6aebb5ea50f112833fa)
repository contains two GGUF files plus a model card, `config.json`,
`params.json`, `SHA256SUMS`, and a license. Use the following trust order:

1. Downloaded bytes verified against the LFS SHA-256 and size.
2. Parsed metadata and tensor inventory from the verified GGUF itself.
3. The pinned official `Qwen/Qwen3.8-27B` configuration and processor files.
4. Human-authored q38rocm documentation only after independent reproduction.

- [ ] Add an automated audit command or test that compares catalog size and
      SHA-256 with both the Hugging Face LFS metadata and `SHA256SUMS`.
- [ ] Explicitly ignore the incorrect sizes in `params.json`:
  - FP4 is understated by `13752800` bytes.
  - FP8 is understated by `11720672` bytes.
- [ ] Add a model-inspection check before runtime installation that verifies:
  - `general.architecture = qwen35`
  - 65 blocks: 64 target layers plus the MTP layer
  - context length 262,144
  - embedding width 5,120
  - FFN width 17,408
  - 24 attention heads and 4 KV heads
  - 248,320 tokenizer entries
  - one next-token-prediction/MTP layer
  - the expected custom GGUF file type and tensor count
- [ ] Reject an artifact that passes SHA validation but fails the expected GGUF
      semantic metadata checks.
- [ ] Record that the model card's sample inspection key
      `qwen3.context_length` is incorrect for these files; the actual key is
      `qwen35.context_length`.
- [ ] Investigate why the GGUF metadata identifies `quantized_by = Unsloth` and
      points `general.repo_url` to Unsloth while the Hugging Face card says
      `quantized_by: julianmb`.
- [ ] Resolve the model card's abbreviated engine revision `e87d53e (213)` to a
      real, immutable full commit containing the required `qwen35`, ROCmFP4,
      ROCmFP8, MTP, and TurboQuant implementations.
- [ ] Record that the model repository contains no raw benchmark artifacts,
      conversion manifest, calibration manifest, build SBOM, or visual
      companion; obtain these from a pinned source or reproduce them locally.
- [ ] Confirm whether the repository's short `LICENSE` file and the official
      base model's Apache-2.0 license provide all required notices and
      attribution for redistribution.
- [ ] Do not ingest the repository's `config.json` or `params.json` as runtime
      configuration; retain them only as audited source records.

### Unsloth reference artifacts

The pinned
[`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/f1bfb127c64f7072bdd2cad55f258b9c8b2910fe)
repository is a useful upstream/reference source. Unlike the q38rocm model
repository, its configuration matches the official Qwen3.8 multimodal
architecture and it includes BF16 language weights, a broad GGUF quantization
ladder, and F16/BF16 visual companion files.

- [ ] Pin Unsloth revision
      `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe` as a reference input.
- [ ] Use its architecture configuration to cross-check the official base and
      actual q38rocm GGUF metadata.
- [ ] Determine and document whether the q38rocm ROCmFP4/FP8 files were
      converted directly from this Unsloth release, another Unsloth artifact,
      or the official safetensors.
- [ ] Add the visual companion candidates to the catalog:
  - `mmproj-F16.gguf`: `927607488` bytes, SHA-256
    `cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e`
  - `mmproj-BF16.gguf`: `931146432` bytes, SHA-256
    `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53`
- [ ] Retain the two-shard Unsloth BF16 GGUF as an optional quality reference,
      not a default deployment artifact:
  - shard 1: `49986159616` bytes, SHA-256
    `b9966e82b7a4d87028b5eae061d578ee826305ebf8baea5bfc6e09bad0ba191f`
  - shard 2: `4671576000` bytes, SHA-256
    `92e3943c4f9bd6292a7bef82369f65fed9bfed088b9df0fb2fa2ce17c9edfa02`
- [ ] Use standard Unsloth Q8_0 as a control when evaluating custom ROCmFP8:
  - `Qwen3.8-27B-Q8_0.gguf`: `29047086048` bytes, SHA-256
    `a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348`
- [ ] Do not catalog the entire quantization ladder by default; download only
      artifacts selected for a defined quality, compatibility, or drafter
      experiment.

### Licensing gate

- [ ] Gate licensing per artifact track: unresolved FastFlowLM, FP8, vision, or
      reference-model terms must not block a license-compliant FP4 baseline.
- [ ] Review and document redistribution and commercial-use terms for:
  - q38rocm source and model artifacts.
  - The Qwen base model.
  - The ROCmFPX fork.
  - FastFlowLM source.
  - FastFlowLM precompiled NPU binaries and kernels.
- [ ] Resolve the distinct conditions in FastFlowLM's
      [`TERMS.md`](https://github.com/ROCm/FastFlowLM/blob/e8582e1a658a48efc94792abbf5c45f00993c05f/TERMS.md)
      before publishing redistributable images.

### Phase 0 exit criterion

- [ ] A clean machine can reproduce the exact FP4 verifier from pinned inputs
      without fetching optional NPU, FP8, vision, BF16, or reference-model
      artifacts, and the intended distribution is license-compliant.

## Phase 1: Integrate the ROCmFPX FP4 GPU baseline

This phase is independently releasable and should be completed before the NPU
bridge.

### Engine and configuration

- [ ] Add `rocmfpx` to the engine image, port, and container maps in
      `lib/halo_ai/cli.py`.
- [ ] Add a pinned ROCmFPX image and related settings to
      `config/halo-ai.env.example`.
- [ ] Refactor the llama-compatible renderer so shared options are not
      duplicated between `llamacpp` and `rocmfpx`.
- [ ] Add ROCmFPX settings and validation for:
  - Device selection, initially `Vulkan0`.
  - GPU layer offload.
  - Flash attention.
  - Batch and micro-batch sizes.
  - CPU thread and polling settings.
  - Separate K/V cache types: `q8_0` and `turbo4`.
  - `spec_draft_n_max`.
  - `spec_draft_p_min`.
  - Strict Qwen MTP mode where supported.
- [ ] Keep target quantization a model property so optional FP8 support can use
      the same engine later without complicating the FP4 baseline.
- [ ] Use the documented `n_max=6`, `p_min=0.60` MTP settings only as an initial
      benchmark/profile candidate, not an unquestioned universal default.
- [ ] Benchmark strict greedy MTP and the documented `n_max=7`, `p_min=0.35`
      deep-speculation setting separately; do not label probabilistic settings
      lossless.
- [ ] Validate every documented environment variable and engine flag against
      the pinned runtime before rendering it:
  - `AMD_VULKAN_ICD=RADV`
  - `VK_ICD_FILENAMES`
  - `HSA_OVERRIDE_GFX_VERSION=11.5.1`
  - `GGML_HIP_ENABLE_UNIFIED_MEMORY=1`
  - `RADV_PERFTEST=gpl,sam,nggc`
  - `-np 1`, `-ctxcp 0`, `-cram 16384`, `--poll 100`
- [ ] Omit Vulkan-irrelevant, obsolete, or driver-version-specific variables
      unless a controlled benchmark demonstrates that they are required.
- [ ] Extend host readiness checks to confirm RADV selects `gfx1151`, exposes
      the required cooperative-matrix capabilities, and actually dispatches
      the ROCmFPX kernels rather than a slow or incompatible fallback.
- [ ] Reject unsupported or unsafe passthrough arguments.

The pinned reference invocation is in q38rocm's
[`run_server.sh`](https://github.com/julianmb/q38rocm/blob/ae3d4640041c92f4b0a012d6bed004e0ad9e66b9/run_server.sh).

### Catalog and profiles

- [ ] Add the exact Qwen3.8 FP4 target artifact to
      `config/models.d/strix-halo.json` as the default acquisition target.
- [ ] Add profile `qwen38-27b-rocmfp4-baseline` with speculation disabled.
- [ ] Add profile `qwen38-27b-rocmfp4-mtp` using the model's GPU MTP head.
- [ ] Define `qwen38-27b-rocmfp8-baseline` and
      `qwen38-27b-rocmfp8-mtp` only as disabled, opt-in GPU profiles after the
      FP4 exit criterion passes; neither profile may activate an NPU service.
- [ ] Do not describe FP8 as improving answer quality until the evaluation
      suite confirms it; the source repository's perplexity claim is useful
      evidence but not a substitute for task-level evaluation.

### Model acquisition

- [ ] Generalize the downloader to support catalog-controlled Hugging Face
      downloads outside the existing Transformers/speech path.
- [ ] Require an immutable repository revision and an allowlist of files.
- [ ] Resolve the selected profile's transitive artifact closure before doing
      any network I/O and fetch only that closure.
- [ ] Make the default FP4 baseline closure exactly the ROCmFPX runtime plus
      `Qwen3.8-27B-ROCmFP4-FAST.gguf`; do not include FP8, an NPU model, vision
      companions, BF16 weights, or benchmark controls.
- [ ] Add an acquisition dry-run that prints every file, source revision,
      expected byte count, destination, and total additional disk use.
- [ ] Require explicit opt-in for each optional artifact class: `fp8`, `npu`,
      `vision`, and `reference`.
- [ ] Reuse already-verified files by SHA-256 across profiles and experiments;
      never duplicate an artifact merely because it has another catalog role.
- [ ] Do not preload optional models during install, update, render, baseline
      start, or baseline test.
- [ ] Support resumable downloads.
- [ ] Verify exact file sizes and SHA-256 values before marking a model ready.
- [ ] Ensure partial or corrupt downloads never replace a valid model.

### Tests

- [ ] Add engine configuration and catalog validation unit tests.
- [ ] Add command-rendering tests for the FP4 baseline and FP4 MTP profiles.
- [ ] Add separate tests proving optional FP8 profiles require explicit
      acquisition and never start an NPU service.
- [ ] Test K/V cache types independently.
- [ ] Test readiness, health URL, lifecycle, image digest, and trial recording.
- [ ] Add command-injection and disallowed-argument tests.
- [ ] Add or extend the smoke test for the new profiles.

### Phase 1 exit criterion

- [ ] The FP4 baseline and FP4 GPU-MTP profiles start through Halo, pass health
      checks, and reproduce direct q38rocm correctness and performance within
      an agreed tolerance without downloading any optional artifact.

### Upstream performance claims to reproduce

Treat these as comparison targets from a self-reported synthetic benchmark,
not as verified Halo performance:

- FP4 raw decode: 14.02 tokens/sec.
- FP4 strict greedy MTP: 34.82 tokens/sec.
- FP4 `n_max=6`, `p_min=0.60`: 30.56-34.82 tokens/sec.
- FP4 `n_max=7`, `p_min=0.35`: reported peak of 36.04 tokens/sec.
- FP8 raw decode: 7.66 tokens/sec.
- FP8 MTP: 18.96 tokens/sec.
- FP8 quality: claimed less than 0.003 perplexity loss, with no supporting
  evaluation artifact in the model repository.
- FP4 at 32K context: approximately 2.45 GiB KV cache, 13.62 raw
  tokens/sec, and 26.85 MTP tokens/sec.

- [ ] Preserve the exact upstream hardware/software description alongside the
      reference values: Ryzen AI Max+ 395, Radeon 8060S, 128 GB
      LPDDR5X-8000, Linux 7.0, and Mesa 26.0 RADV.
- [ ] Reproduce the FP4 claims first through Halo with raw benchmark output,
      exact prompts, sampling settings, build IDs, and system telemetry.
- [ ] If the optional FP8 track is explicitly authorized later, measure its
      context-memory scaling independently and reproduce its perplexity claim
      on a named, versioned corpus; do not fetch BF16 or Q8 controls merely to
      complete the FP4 baseline.
- [ ] Add preflight memory estimates for weights, KV cache, runtime buffers,
      visual companion, and NPU drafter before allowing large contexts.
- [ ] Report median and p95 over repeated runs; do not promote a peak
      single-run number as the profile expectation.

## Optional parallel track: Restore native vision-language support

The official model card and configuration describe a native vision-language
model with a 27-layer vision encoder, image/video token IDs, a
`Qwen3VLProcessor`, and a `Qwen3_5ForConditionalGeneration` architecture. The
audited q38rocm FP4 and FP8 GGUFs contain the matching 248,320-token `qwen35`
language decoder and MTP layer, but contain no visual tensors. The q38rocm
repository also provides no companion projector. Until one is added and
validated, those GGUF files alone are text-only deployments of a multimodal
base model.

Vision restoration is optional and must not add a companion download to the
text-only FP4 baseline:

- [ ] Add explicit `modalities` metadata to models and profiles; default to
      `text` and never infer vision support from quantization.
- [ ] Verify the pinned ROCmFPX/llama.cpp fork's current Qwen3.5 vision support,
      including image preprocessing, projector conversion, and video support.
- [ ] Test Unsloth's pinned `mmproj-F16.gguf` and `mmproj-BF16.gguf` against
      the FP4 language GGUF first; evaluate FP8 separately only if its optional
      GPU profile has already been authorized and acquired.
- [ ] Verify projector metadata, source revision, tensor shapes, output width,
      visual token layout, preprocessing, and image/video special-token IDs.
- [ ] If either existing companion is not demonstrably compatible, produce a
      new visual encoder/projector from the exact pinned `Qwen/Qwen3.8-27B`
      revision using a pinned conversion toolchain.
- [ ] Use BF16 as the quality reference and prefer F16 for the normal profile
      only if outputs and visual evaluation remain within tolerance; evaluate a
      lower-bit projector only after correctness is established.
- [ ] Pin and verify the projector, preprocessor, video preprocessor,
      tokenizer, special token IDs, and chat template.
- [ ] Prove that the projector output dimension and visual token layout match
      the FP4 language GGUF; do not make FP8 validation a prerequisite.
- [ ] Extend renderer validation for the engine's vision/projector arguments.
- [ ] Extend the downloader and catalog schema for typed companion artifacts.
- [ ] Verify OpenAI-compatible image inputs, local-file policy, remote-image
      policy, image size limits, and multiple-image requests.
- [ ] Add image-captioning, OCR, chart, document, and visual-reasoning tests.
- [ ] Benchmark vision encoder time, TTFT, text generation, memory use, and
      context growth separately.
- [ ] Keep external NPU drafting disabled for every image/video request; an
      NPU-drafted multimodal profile is outside the current scope.
- [ ] Measure target-native MTP first because it remains conditioned on the
      verifier's visual hidden state.
- [ ] Prefer target-native MTP for the first vision profile if it is supported,
      since it can remain conditioned on the target's multimodal hidden state.

### Vision exit criterion

- [ ] The official Qwen3.8 vision stack processes images through the FP4
      verifier correctly, with the separately acquired visual companion and
      supported modalities explicitly recorded in the catalog.

## Phase 2: Prove Vulkan and XDNA2 coexistence

Do not build the speculative bridge until this phase passes.

### Host readiness

- [ ] Extend `halo-ai host doctor` to verify:
  - Firmware IOMMU is enabled.
  - The kernel command line does not contain `amd_iommu=off` in NPU mode.
  - `amdxdna` is loaded and bound to the expected device.
  - `/dev/accel/accel0` exists and is accessible to the runtime user.
  - XRT and NPU firmware versions are supported and mutually compatible.
  - Render/accel group permissions are correct.
  - Locked-memory limits are sufficient.
  - Vulkan selects the intended Radeon device.
  - `xrt-smi` can query and exercise the NPU.
- [ ] Run the pinned FastFlowLM `flm validate` check and retain its structured
      result in diagnostics.
- [ ] Optionally compare Halo's result with the pinned Lemonade backend-status
      check, but do not make Lemonade installation a prerequisite for direct
      FastFlowLM drafting.
- [ ] Replace q38rocm's `"lemonade" in groups or "render" in groups`
      heuristic with checks of the actual `/dev/accel/accel0` ownership, mode,
      ACL, effective process groups, and open/ioctl access.
- [ ] Make NPU host mode refuse readiness until every mandatory check passes.
- [ ] Preserve Halo's existing GTT and memory tuning when switching host mode.
- [ ] Document the reboot boundary and how to return to GPU-only mode.

### Hardware feasibility matrix

- [ ] Benchmark the Vulkan verifier alone.
- [ ] Benchmark a FastFlowLM NPU model alone.
- [ ] Load both engines and verify idle stability and memory use.
- [ ] Generate concurrently for at least 30 minutes.
- [ ] Repeat concurrent generation with long-context verifier load.
- [ ] Capture:
  - Kernel and journal errors.
  - XRT and firmware timeouts or resets.
  - GPU faults.
  - Memory and locked-memory pressure.
  - Temperature, clocks, and power.
  - Throughput degradation for both engines.

Related upstream reports include
[`amd/xdna-driver#1605`](https://github.com/amd/xdna-driver/issues/1605),
which describes failures during concurrent NPU and ROCm/HIP activity. The
target runtime uses Vulkan, so this exact pairing must be measured.

### Phase 2 stop/go gate

- [ ] **Go:** concurrent Vulkan and NPU execution is repeatably stable with
      tolerable contention.
- [ ] **Stop:** if the combination resets, hangs, or suffers persistent
      performance collapse, ship Phase 1 and keep NPU support disabled.

## Phase 3: Produce a tiny, same-family Qwen3.8 NPU draft model

The currently available models cannot be assumed compatible:

- The official target and both audited q38rocm GGUFs use a 248,320-token
  vocabulary. The q38rocm companion config's 152,064 value is stale or
  incorrect and must not be used.
- FastFlowLM's
  [`Qwen3-0.6B-NPU2`](https://huggingface.co/FastFlowLM/Qwen3-0.6B-NPU2)
  declares 151,936 tokens and is not a drop-in draft model.
- The available
  [`Qwen3.8-1.0B-A0.6B`](https://huggingface.co/inference-optimization/Qwen3.8-1.0B-A0.6B)
  uses a small Qwen3.5-MoE architecture and the correct 248,320-token
  vocabulary. Its model card says it was dimension-reduced from the
  Qwen3.8-2.4T-A95B MoE checkpoint and fine-tuned only on a toy dataset for
  testing. It is therefore a useful converter/kernel fixture, **not a
  production drafter or a quality proxy for the 27B target**.
- Unsloth's 27B UD-IQ2/3 files preserve the exact target architecture and
  weights at aggressive quantization. They are useful acceptance and quality
  references, but their iMatrix-derived GGUF encodings are **not directly
  compatible with FastFlowLM** and they remain full 27B models.

Using the same model family is a hard requirement. Tokenizer compatibility is
necessary but not sufficient: the draft model must retain Qwen3.8's prompt
format, special tokens, thinking behavior, and learned output distribution so
that acceptance remains high. A generic Qwen3 model is not an eligible release
drafter.

Two same-family candidate classes should be compared where the selected NPU
runtime can actually execute them:

1. Exact-target 27B weights in a runtime-native NPU format, maximizing model
   agreement but retaining the target's layer and compute count. For
   FastFlowLM this means a separately produced Q4NX artifact, not an Unsloth
   iMatrix GGUF.
2. A tiny Qwen3.8-family model, minimizing NPU proposal latency but potentially
   reducing acceptance.

### FastFlowLM format compatibility gate

The current pinned FastFlowLM loaders open `model.q4nx` through `Q4NX`, which
inherits its SafeTensors loader. Its model manifests pair those weights with
model-specific XDNA `xclbin` kernels. The repository has no GGUF/iMatrix/IQ
loader. Similar-looking quantization labels such as `Q4_1` do not imply GGUF
wire-format compatibility.

| Artifact | FastFlowLM direct load | Native `ggml-xdna` research path | Intended role |
| --- | --- | --- | --- |
| Unsloth UD/IQ iMatrix GGUF | No | Experimental; only supported tensor types, then dequantized/requantized for current int8 AIE kernels | Quality/acceptance reference only |
| Qwen3.8-family SafeTensors | No; Q4NX conversion required | No | Preferred source for producing an NPU artifact |
| `model.q4nx` plus matching Qwen3.8 `xclbin` files | Yes, after adding/validating model support | No | Production FastFlowLM draft artifact |

- [ ] Enforce `artifact_format`, `quantization_method`, `runtime_compatibility`,
      and `required_kernels` in the catalog; reject a GGUF assigned to the
      `fastflow-npu` engine before any service starts.
- [ ] Pin and audit the Q4NX converter, input schema, calibration method,
      supported Qwen3.8 operators, binary dependencies, and redistribution
      terms.
- [ ] Produce Q4NX from pinned SafeTensors rather than dequantizing an iMatrix
      GGUF. The Q4NX result is an independent quantization artifact with a new
      digest and independent quality/acceptance measurements.
- [ ] Build or obtain model-specific Qwen3.8 XDNA kernels; a valid `model.q4nx`
      alone is insufficient.
- [ ] Add a negative test proving that each Unsloth UD/IQ GGUF is rejected by
      the FastFlowLM provider with an actionable format error.
- [ ] Keep native GGUF-on-XDNA as a separate research backend. Test every
      selected IQ tensor type for dequantization support and disclose the
      current int8 re-quantization step; do not describe that path as direct
      execution of the original iMatrix quant.
- [ ] Reject any candidate or profile that silently falls back to CPU or
      reports XDNA without device-counter evidence.

### Qwen3.6-MoE bootstrap experiment

FastFlowLM v0.9.45 added `Qwen3.6-35B-A3B`, v0.9.46 improved its performance,
and the support remains in v1.0.1. This is a valuable way to exercise the
draft protocol against an already-shipping Qwen hybrid-MoE NPU engine before a
Qwen3.8 Q4NX artifact exists. It is an integration fixture only: Qwen3.6 is
not the exact Qwen3.8 target family and must never become the release drafter.
Because Qwen3.8 is a newer post-training generation, assume cross-version
token agreement is poor until a direct benchmark proves otherwise; shared BPE
IDs and similar decoder operators are not evidence of useful acceptance.

- [x] Pin the bootstrap model revision and inspect its artifact manifest:
  - Repository: `FastFlowLM/Qwen3.6-35B-A3B-NPU2`
  - Revision: `0cad6285baf6f37adf2c4e9696372c0140078fe0`
  - `model.q4nx`: `23235412536` bytes, SHA-256
    `688f1e153d10ad0bc2d9ab6a4931cf3d91e9f26c8f817ebb8a2d60a56a3cf8de`
  - Optional `vision_weight.q4nx`: `1008858888` bytes, SHA-256
    `11845c3c112ccb481382763e7e1838cca103b100a73bbd98b7a61cc12861323e`
  - Declared runtime footprint: 24.3 GB
  - Quantization: FastFlowLM `Q4_K_S`/NPU2, not GGUF
- [x] Record the architectural overlap with the proposed Qwen3.8 MoE fixture:
  both use the `Qwen3_5Moe` hybrid decoder primitives and repeat three linear
  attention layers followed by one full-attention layer. Dimensions, layer
  count, expert count/routing, head dimensions, and tensor shapes differ, so
  the Qwen3.6 binary and `xclbin` set cannot be reused unchanged for the tiny
  model.
- [x] Perform a static tokenizer comparison against the pinned Qwen3.8 target:
  - BPE vocabulary, merge table, normalization, pre-tokenization,
    post-processing, and decoding metadata match semantically.
  - Token IDs `0..248069`, including chat, vision, tool, FIM, and thinking
    tokens, agree.
  - Qwen3.6 omits Qwen3.8's audio special tokens `248070..248076`; therefore
    this is only a text-only compatibility experiment, not exact tokenizer
    identity.
- [x] Confirm that the internal Qwen3.6 engine already exposes the primitives
  needed by a draft adapter: token-ID `prefill`, one-token `forward`,
  `checkpoint`, `restore`, and `clear_context`. The public HTTP API still does
  not expose the required stateful token protocol.
- [x] Apply a performance pre-screen using published figures. Qwen3.6 NPU
  decode is 13.65 tok/s at 1K and 9.51 tok/s at 32K. The q38rocm repository
  reports 14.02 tok/s raw FP4 and 30.56--36.04 tok/s FP4 MTP. Qwen3.6 is
  therefore unlikely to accelerate FP4; test it for protocol and acceptance
  behavior, not as a presumed performance win.
- [ ] Before downloading Qwen3.6 Q4NX, run a no-new-model-download version-mix
      screen using an already-installed, hash-verified Qwen3.6-35B-A3B GPU
      artifact if one is present. If it is not already present, skip this proxy
      rather than downloading another Qwen3.6 representation solely for the
      screen.
- [ ] Use the FP4 baseline as the sole verifier and feed both models the exact
      Qwen3.8-rendered token prefix under greedy decoding; never compare
      independently rendered Qwen3.6 and Qwen3.8 prompts.
- [ ] Measure first-token agreement, zero-acceptance frequency, mean accepted
      tokens per proposal, accepted-run p50/p95, and divergence position for
      proposal lengths 1, 2, 4, and 6 across code, reasoning, JSON/tool use,
      multilingual text, ordinary chat, and thinking/non-thinking prompts.
- [ ] Record agreement by prompt class and context bucket; do not hide a poor
      domain behind an aggregate acceptance percentage.
- [ ] Model end-to-end throughput from the measured acceptance distribution,
      published NPU proposal latency, and measured FP4 verification latency.
      Treat the version mix as failed if it cannot plausibly beat FP4 GPU MTP.
- [ ] Download the Qwen3.6 Q4NX artifact only after the version-mix screen is
      promising, the token-level adapter is ready to exercise it, and the
      operator explicitly selects the `npu` experiment.
- [ ] On a supported Strix Halo Linux host, pin FastFlowLM v1.0.1, run
      `flm validate`, and only then pull and checksum
      `qwen3.6-moe:35b-a3b`:
  - Source commit: `2af137496db8aa27a325b5335044b62dfd2b6ea2`
  - CachyOS portable asset: `fastflowlm_1.0.1_linux.tar.gz`
  - Portable asset SHA-256:
    `bd3936cb2c8b099a4b90b4d20bd8f62ae6693a5420e3ae095951201b2bfbd1ca`
- [ ] Determine whether text-only Qwen3.6 can run without downloading
      `vision_weight.q4nx`; if FastFlowLM requires it, include the extra byte
      cost in the acquisition dry-run and do not bypass loader validation.
- [ ] Run text-only, greedy `flm bench` at 1K, 4K, and 32K context while
      recording XRT device counters, memory, power, and any CPU fallback.
- [ ] Add a minimal internal-API probe that accepts already-tokenized prompt
      IDs, performs greedy proposals of lengths 1, 2, 4, and 6, and reports
      token IDs rather than decoded text.
- [ ] Exercise checkpoint/restore with accepted-all, accepted-none, and
      partially accepted proposals for at least 10,000 cycles under ASan/UBSan
      where the host libraries permit it.
- [ ] Send target-rendered Qwen3.8 text prompt IDs directly to the bootstrap
      drafter; do not apply the Qwen3.6 chat template or re-tokenize text in the
      draft daemon.
- [ ] Refuse bootstrap requests containing image, video, audio, or token IDs
      `248070..248076`.
- [ ] Measure greedy token agreement and accepted-run distribution against
      q38rocm FP4 on the same pinned corpus used by the download-free proxy.
- [ ] Run a bounded end-to-end speculative test only after the token/KV probe
      passes. Compare against raw target decoding and target-native MTP, and
      stop the experiment if proposal latency cannot yield a modeled gain.
- [ ] Keep this bootstrap profile test-only and unavailable in the normal Halo
      catalog, regardless of measured acceptance.

### Family and lineage conformance

- [ ] Require a documented Qwen3.8 base-model lineage in the draft artifact's
      immutable metadata.
- [ ] Verify architecture family, tokenizer origin, chat template, generation
      defaults, thinking controls, and special-token semantics against the
      official Qwen3.8 target.
- [ ] Record `model_family`, `base_model`, `base_revision`, architecture,
      parameter count, active parameter count, and training/distillation
      provenance in the catalog.
- [ ] Reject generic Qwen3, Qwen3.5, or unrelated models even if their token ID
      maps happen to be compatible.
- [ ] Permit dense/MoE and layer-count differences only within an explicitly
      documented Qwen3.8 small-model or distilled-model lineage.

### Tokenizer conformance

- [ ] Store the FP4 target tokenizer fingerprint; compare optional FP8 metadata
      to it without making the FP8 weights a required download.
- [ ] Compare the target and draft `tokenizer.json` files.
- [ ] Compare every shared token string and numeric ID.
- [ ] Compare normalization, pre-tokenization, added tokens, special tokens,
      and chat templates.
- [ ] Hash the canonical tokenizer definition and store the fingerprint in the
      model catalog.
- [ ] Test arbitrary text round trips and a corpus containing every special
      token.
- [ ] Determine whether every token the target can emit can be consumed by the
      drafter.
- [ ] Refuse startup when tokenizer compatibility is not proven.

### Draft artifact

- [ ] Retain this bounded exact-target set from the pinned Unsloth repo as
      GGUF quality/acceptance references and optional native-XDNA experiments,
      **not FastFlowLM draft artifacts**:
  - `Qwen3.8-27B-UD-IQ2_XXS.gguf`: `9010048064` bytes, SHA-256
    `8d1b37297d6cf98303cd396896f35e01089ddcc904053a9c6997f7a1c35b8524`
  - `Qwen3.8-27B-UD-IQ2_M.gguf`: `10319907904` bytes, SHA-256
    `04a89ef4fa9c8726d09331433346809bbab692b4851d49d0738ba8d58a1ae740`
  - `Qwen3.8-27B-UD-IQ3_XXS.gguf`: `11913559104` bytes, SHA-256
    `0a6129dcbbbe72f423dc67e0e3bbfbbdf3e923981a3637687ebb96a46c59d6be`
- [ ] Use `Qwen3.8-1.0B-A0.6B` only as a Qwen3.8-MoE converter and kernel
      bring-up fixture: approximately 1.0B total and 0.6B active parameters,
      revision `36531b2b4eab12e9cf7b7b9cbfce0f0a0bb8b719`. Do not measure its
      toy-trained output as representative draft quality.
- [ ] Verify every tokenizer ID and special token; equal vocabulary size alone
      is insufficient.
- [ ] For the optional native `ggml-xdna` spike, determine which exact-target
      GGUF tensor types expose the required dequantization path and measure the
      cost and quality impact of conversion to the AIE kernel's int8 weights.
- [ ] If evaluating an exact-target 27B FastFlowLM drafter, start from the
      pinned official or provenance-verified SafeTensors and produce Q4NX
      directly; do not use an iMatrix GGUF as the conversion source.
- [ ] Add or adapt FastFlowLM support for the candidate's Qwen3.5-MoE decoder.
- [ ] If the candidate is unsuitable, select, distill, or train another tiny
      Qwen3.8-family model using the exact official tokenizer and semantics.
- [ ] Require a production candidate to have meaningful public evaluation or
      reproduce its training/distillation and evaluate it before spending
      time on Q4NX/kernel optimization.
- [ ] Optionally distill the drafter on target-generated responses to improve
      acceptance.
- [ ] Establish an unquantized or highest-supported-precision reference before
      quantizing so quality and acceptance loss can be measured.
- [ ] Evaluate NPU-native quantization candidates supported by the chosen XDNA
      kernels rather than assuming that the lowest bit width is fastest.
- [ ] Select the smallest quantization on the latency/acceptance Pareto frontier
      that fits the NPU comfortably and improves end-to-end throughput.
- [ ] Include weights, KV cache, activation buffers, XRT allocations, kernels,
      and fragmentation in the NPU memory budget.
- [ ] Reserve explicit memory headroom for maximum supported draft context and
      repeated-session operation.
- [ ] Convert the selected SafeTensors source to a reproducible `model.q4nx`
      using the audited Q4NX recipe.
- [ ] Build all required model-specific XDNA `xclbin` kernels.
- [ ] Compare reference and Q4NX logits/token choices before measuring
      speculative acceptance and end-to-end throughput.
- [ ] Pin the model, tokenizer, converter, calibration/distillation data
      manifest, quantization recipe, and kernels in the catalog.

### Drafter benchmark

- [ ] Measure prompt ingestion speed.
- [ ] Measure single-token decode speed.
- [ ] Measure proposal batches of 2 through 8 tokens.
- [ ] Measure rollback and target-token resynchronization cost.
- [ ] Compare draft precisions using NPU memory, cold-load time, prefill,
      decode, proposal latency, energy, and acceptance—not model size alone.
- [ ] Compare exact-target aggressive quants with the small active-parameter
      model on the same prompts, contexts, proposal lengths, and target
      quantizations.
- [ ] Reject a 27B draft candidate if its NPU proposal latency erases the
      verifier-side gain, regardless of its higher acceptance rate.
- [ ] Measure acceptance against the Qwen3.8 target for representative tasks.
- [ ] Measure acceptance only against the pinned FP4 target. FP8 is not an NPU
      draft target in this plan.
- [ ] Use the measured verifier latency, proposal latency, and acceptance to
      predict end-to-end speculative throughput.
- [ ] Verify NPU execution using XRT/device counters rather than process names
      or claimed backend strings.

### Phase 3 stop/go gate

- [ ] **Go:** the drafter is stable, genuinely runs on XDNA2, consumes the
      target token stream safely, passes same-family conformance, fits with
      operational headroom, and predicts a meaningful gain over GPU MTP.
- [ ] **Stop:** retain the model experiments, but do not add production Halo
      plumbing if proposal latency or acceptance cannot beat GPU MTP.

## Phase 4: Implement the token-level NPU draft provider

FastFlowLM is the preferred production path because it supplies optimized
full-model XDNA2 kernels. Its current HTTP generation API does not expose the
stateful token operations required by speculative decoding. The relevant
upstream request is
[`ROCm/FastFlowLM#588`](https://github.com/ROCm/FastFlowLM/issues/588).

### Lemonade integration decision

The `lemonade` reference in q38rocm's `strix_doctor.py` is only the name of a
Unix group accepted by its permission heuristic. The script does not call
Lemonade, load a Lemonade model, or verify NPU inference.

The official
[`lemonade-sdk/lemonade`](https://github.com/lemonade-sdk/lemonade/tree/7a960ed3850d0bb92e0e96b577bf4e9505612172)
runtime is nevertheless relevant: it supports a FastFlowLM `flm` backend on
XDNA2 under Windows and Linux, can install/pin FLM, runs `flm validate`,
discovers FLM models, and manages the backend process. Its current FastFlowLM
wrapper launches `flm serve` and forwards ordinary chat/completion requests.
Its declared modes are chat, embeddings, and transcription; it does not expose
a speculative-draft capability, persistent proposal session, KV checkpoint or
rollback, or verifier synchronization protocol.

Current decision: Halo should launch a purpose-built FastFlowLM draft daemon
and connect the ROCmFPX verifier directly over a private Unix socket. Do not
chain OpenAI-compatible requests through Lemonade for every proposal. That
would add HTTP/proxy overhead and, more importantly, would not provide the
stateful token/KV operations required for efficient speculation.

- [ ] Reuse or adapt Lemonade's XDNA2 detection, `flm validate` interpretation,
      backend-version discovery, and Linux troubleshooting knowledge where it
      reduces duplicated maintenance.
- [ ] Pin FastFlowLM independently in Halo; do not inherit Lemonade's
      `VersionPolicy::AtLeast` behavior for a correctness-sensitive draft
      runtime.
- [ ] Keep model/artifact verification in Halo even if Lemonade or FLM performs
      its own download.
- [ ] Measure direct FastFlowLM draft-daemon startup and operation before adding
      a second supervisor/router layer.
- [ ] Do not send per-proposal OpenAI chat or completion calls to Lemonade.
- [ ] Consider upstreaming a first-class `draft` capability to Lemonade after
      the FastFlowLM protocol works, including session, proposal, resolve,
      health, and telemetry operations.
- [ ] If Lemonade later supports the exact draft protocol, immutable backend
      pinning, and direct socket handoff without proxying the hot path, reassess
      whether it should manage the drafter process.

### Drafter protocol

- [ ] Write a small protocol specification before implementation.
- [ ] Use a private Unix-domain socket with versioned, length-prefixed messages.
- [ ] Define these operations:
  - `HELLO`: protocol version, model ID, Qwen3.8 lineage, quantization,
    tokenizer fingerprint, backend, XRT, firmware, and kernel versions.
  - `OPEN`: create a session and prefill prompt token IDs.
  - `PROPOSE`: return up to N greedy token IDs plus timing data.
  - `RESOLVE`: provide the accepted count and verifier-selected correction
    token, rolling back the rejected suffix.
  - `RESET`: replace a session's prefix state.
  - `CLOSE`: release session resources.
  - `HEALTH`: liveness, readiness, and actual device/backend state.
  - `STATS`: proposal, acceptance, timing, error, and NPU counters.
- [ ] Specify maximum message sizes, integer widths, error codes, timeouts, and
      backward-compatibility rules.

### FastFlowLM adapter

- [ ] Add a small FastFlowLM drafter executable using the internal model API,
      not its text-generation HTTP endpoint.
- [ ] Expose token-ID prefill and greedy generation.
- [ ] Add KV-cache checkpoints or rollback for rejected proposal suffixes.
- [ ] Append the verifier-selected correction token without replaying the full
      prompt.
- [ ] Bound session memory, request size, and concurrency.
- [ ] Return verifiable NPU backend and device telemetry.
- [ ] Add unit tests for session transitions and KV synchronization.

### ROCmFPX verifier adapter

- [ ] Add a `remote-draft` provider to the native speculative coordinator.
- [ ] Keep sampling, acceptance, and emitted-token decisions inside the
      verifier process.
- [ ] Implement strict greedy speculation first.
- [ ] Defer stochastic speculation until the drafter can return the probability
      distributions needed for correct rejection sampling.
- [ ] Add bounded connection, proposal, and health deadlines.
- [ ] Add backpressure and a maximum number of in-flight sessions.
- [ ] Make a lost, stale, or incompatible drafter fail the NPU profile.
- [ ] Never silently redirect drafting to CPU or GPU.

### Secondary research path

- [ ] Track
      [`OllamaAMDNPU`](https://github.com/BrandedTamarasu-glitch/OllamaAMDNPU)
      as a correctness reference for a native ggml XDNA backend.
- [ ] Reconsider porting `ggml-xdna` into ROCmFPX only if its measured decode
      performance becomes competitive.
- [ ] Keep Qwen3.8 MTP-head-only NPU offload as later research; do not block the
      full-model drafter path on it.

### Phase 4 exit criterion

- [ ] A standalone verifier/drafter pair performs strict greedy speculative
      decoding, maintains synchronized KV state, and survives rejection and
      failure-injection tests.

## Phase 5: Add multi-service runtimes to Halo

### Catalog and schema

- [ ] Generalize a profile from one container to a logical runtime containing
      ordered services.
- [ ] Preserve backward compatibility for existing single-service profiles.
- [ ] Add first-class catalog relationships for:
  - `target_model`
  - `draft_model`
  - `model_family` and base-model lineage
  - tokenizer fingerprint
  - artifact format and quantization method
  - allowed runtime/provider and required kernel set
  - quantization and NPU memory budget
  - NPU kernel artifacts
  - declared compatibility and protocol version
- [ ] Represent drafting as an optional block on the FP4 profile without
      duplicating verifier engine logic.
- [ ] Validate every referenced artifact and relationship before rendering a
      runtime.

### Engines and profiles

- [ ] Add `fastflow-npu` as an internal engine/service.
- [ ] Add profile `qwen38-27b-rocmfp4-npu-draft`.
- [ ] Do not add an FP8+NPU profile; reject that combination in catalog
      validation if it is declared accidentally.
- [ ] Mark the FP4 NPU profile experimental and disabled unless the host,
      artifact, compatibility, and performance gates pass.
- [ ] Keep the baseline and GPU MTP profiles directly selectable; do not use an
      implicit mid-request fallback.

### Transactional lifecycle

- [ ] Create a private per-runtime directory and socket.
- [ ] Start the drafter first.
- [ ] Validate its handshake, model, tokenizer, backend, and real NPU use.
- [ ] Validate same-family lineage, quantization, and declared NPU memory budget
      during the handshake before starting the verifier.
- [ ] Start the verifier only after the drafter is ready.
- [ ] Roll back both services and runtime files if either startup fails.
- [ ] Treat both containers as one unit for start, stop, status, cleanup,
      update, and trial commands.
- [ ] Detect and clean stale sockets without deleting unrelated user files.
- [ ] Record both image digests and all model/kernel revisions in trial state.

### Isolation and security

- [ ] Keep both services rootless.
- [ ] Mount models, tokenizers, and NPU kernels read-only.
- [ ] Give only the drafter `/dev/accel/accel0`.
- [ ] Give only the verifier the required Vulkan/render devices.
- [ ] Share a mode-`0600` Unix socket in the private runtime directory.
- [ ] Give the drafter no network access.
- [ ] Expose only the verifier API and bind it to `127.0.0.1` by default.
- [ ] Add least-privilege and mount-validation tests.

### Observability

- [ ] Record proposed tokens, accepted tokens, acceptance rate, and accepted
      run-length distribution.
- [ ] Record drafter tokens/sec and proposal latency.
- [ ] Record verifier batch time and end-to-end tokens/sec.
- [ ] Record backend identities, XRT counters, NPU faults, and firmware resets.
- [ ] Record fallback count and require it to remain zero.
- [ ] Include both service logs in diagnostics while avoiding prompt-content
      leakage by default.

### Phase 5 exit criterion

- [ ] Halo can safely install, validate, start, monitor, stop, and update the
      two-service runtime as one profile.

## Phase 6: Correctness, reliability, and performance qualification

### Hardware-independent tests

- [ ] Add schema, catalog, renderer, and protocol unit tests.
- [ ] Add a fake drafter for CI so the full speculative state machine is tested
      without AMD hardware.
- [ ] Test accepted-all, accepted-none, partial acceptance, correction-token,
      reset, and close transitions.
- [ ] Test mismatched tokenizers, models, kernels, protocols, and device claims.
- [ ] Test malformed replies, oversized messages, timeouts, crashes, and
      disconnects.
- [ ] Test transactional cleanup after failures at every startup stage.

### Correctness corpus

- [ ] Compare FP4 NPU-drafted strict greedy output token-for-token with
      unassisted FP4 target decoding for:
  - Code generation.
  - Reasoning and mathematics.
  - JSON and tool-shaped output.
  - Multilingual text.
  - Special-token boundaries.
  - Short and long prompts up to at least 32K context.
- [ ] Use fixed prompts, seeds, parameters, software revisions, and expected
      token sequences.
- [ ] Include repeated-prefix and multi-turn/session-resynchronization cases.
- [ ] If the optional FP8 GPU-only track is authorized, compare FP4 and FP8 task
      quality using a fixed evaluation suite covering code, reasoning,
      instruction following, structured output, retrieval, and long-context
      behavior.
- [ ] Report quality deltas and confidence intervals rather than relying only
      on perplexity or anecdotal prompts.
- [ ] Do not require optional FP8 to emit the same tokens as FP4; FP8 has no
      external drafted profile in this plan.

### Reliability

- [ ] Inject drafter crashes and timeouts during prefill and proposal.
- [ ] Inject verifier shutdowns while sessions exist.
- [ ] Verify no orphaned containers, sockets, or accelerator contexts remain.
- [ ] Run a multi-hour concurrent Vulkan/NPU soak.
- [ ] Check kernel logs and XRT telemetry automatically after the soak.
- [ ] Verify repeated start/stop cycles do not degrade either accelerator.

### Performance

- [ ] Benchmark cold and warm runs for:
  - Unassisted ROCmFP4.
  - ROCmFP4 GPU MTP.
  - ROCmFP4 with NPU drafting.
  - Optionally, and only after explicit acquisition: unassisted ROCmFP8 and
    ROCmFP8 GPU MTP.
- [ ] Cover code, reasoning, structured output, ordinary chat, and long-prefix
      workloads.
- [ ] Report median and p95 TTFT, generation latency, and tokens/sec.
- [ ] Report proposal length, acceptance, draft latency, verification latency,
      memory, temperature, and power.
- [ ] Tune proposal length only from recorded measurements; do not bake the
      theoretical q38rocm estimates into defaults.

### Promotion gate

- [ ] Exact greedy token equivalence passes.
- [ ] Actual NPU execution counters are nonzero.
- [ ] Silent fallback count is zero.
- [ ] No driver, firmware, GPU, or NPU faults occur during qualification.
- [ ] Long-context operation remains stable.
- [ ] FP4 NPU-drafted median throughput is at least 10% above the FP4 GPU MTP
      profile on the representative corpus.
- [ ] p95 latency and power satisfy the agreed release limits.

## Phase 7: Documentation, release, and upstreaming

- [ ] Document host preparation, firmware and XRT compatibility, reboot
      requirements, and recovery procedures.
- [ ] Document profile selection and explain when GPU MTP is preferable.
- [ ] Clearly label NPU drafting experimental until every promotion gate passes.
- [ ] Publish the draft protocol specification and compatibility matrix.
- [ ] Document independent upgrade and rollback procedures for:
  - Halo
  - ROCmFPX verifier
  - FastFlowLM drafter
  - amdxdna/XRT/firmware
  - target and draft models
  - NPU kernels
- [ ] Publish reproducible benchmark inputs and reports.
- [ ] Upstream the generic FastFlowLM session/draft API where practical.
- [ ] Upstream the generic ROCmFPX/llama.cpp remote-draft provider where
      practical.
- [ ] Promote the NPU profile only after the qualification criteria pass.

## Suggested implementation slices

Each slice should be independently reviewable and leave the existing runtime
working:

1. [ ] FP4-baseline architecture decision, provenance manifest, and licensing
       report; record optional-track licensing separately.
2. [ ] ROCmFPX image, engine, FP4 catalog entry, acquisition dry-run, and
       unassisted FP4 baseline profile.
3. [ ] FP4 GPU MTP profile and reproducible correctness/performance report.
4. [ ] Qwen3.6-to-Qwen3.8 version-mixing acceptance screen using already cached
       artifacts where possible.
5. [ ] NPU host doctor and Vulkan/XDNA coexistence report.
6. [ ] Qwen3.8 family/tokenizer conformance tool and tiny-quant drafter
       decision.
7. [ ] Standalone FastFlowLM token drafter and protocol tests.
8. [ ] ROCmFPX remote-draft provider and correctness harness.
9. [ ] Halo multi-service lifecycle and experimental FP4 NPU profile.
10. [ ] Hardware qualification, documentation, and promotion decision.
11. [ ] Optional FP8 GPU-only catalog/profile and quality work, separately
        authorized and downloaded only on demand.
12. [ ] Optional FP4 vision-language target, separately acquired projector,
        and multimodal qualification.
