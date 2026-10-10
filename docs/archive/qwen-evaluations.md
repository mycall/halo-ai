# Qwen profiles and runtime experiments

> Archived evidence. Settings and host state describe the original trials.
> Use the [reference manual](../README.md) for supported operations.

[Documentation index](../README.md) · [Project README](../../README.md)

For profile selection and the acquisition workflow, start with [Profiles](../profiles.md).
The recommended Flash-Next setup has its own [Halogen guide](../halogen-flash.md).
This guide covers the other Qwen runtimes, compatibility limits, and qualification evidence.

- [27B aliases, vision, DFlash2, and OpenCode](qwen-evaluations.md#27b-aliases-vision-dflash2-and-opencode)
- [Flash-Next GGUF and MTP](qwen-evaluations.md#qwen38-flash-next-ud-q4_k_xl-and-mtp)
- [ROCmFP4 and FP8](qwen-evaluations.md#qwen38-rocmfp4-baseline)

## 27B aliases, vision, DFlash2, and OpenCode

The Qwen3.8 lanes have explicit short aliases:

```bash
halo-ai start qwen3.8-fn --switch  # Recommended Flash-Next: Halogen + MTP + vision, 262K
halo-ai start qwen3.8-27b --switch  # Balanced 27B default: Q6 XL + vision + Q8 DFlash2, 65K
halo-ai start qwen3.8df2 --switch  # Q6 XL vision + DFlash2, native 262K
halo-ai start qwen3.8-27b-q6xl-vision-lemonade --switch  # Same Q6/BF16 target on ROCm/HIP
halo-ai start qwen3.8fp4 --switch
halo-ai start qwen3.8fp8 --switch
```

`qwen3.8-27b` is the balanced 27B family default. It resolves to the qualified
v0.7.5 Q6 XL vision profile with Q8_0 DFlash2 at 65K. `qwen3.8df2` preserves
the native-262K route and uses the separately pinned older Strix Vulkan fork, not
ROCmFPX. It aliases the experimental vision+DFlash2 profile with the Q4_K_M
drafter at `n-max=5`. A paired structured-image canary matched target-only in
three fresh processes while decoding at 24.53 tok/s versus 8.43 target-only.
The equal-history fixed suite also matched all 13 output-token streams in two
fresh-process repetitions. Use `qwen3.8-27b-q6xl-strix-vision` for the
target-only rollback.
Three fresh-process repetitions also tested the official Q8_0 and BF16
DFlash2 drafters at `n-max=7`. They did not resolve the observed divergence and
were slower than Q4_K_M in that canary, but this does not establish that Q4 is
generally more faithful: the later block-size sweep identified verification
batch shape as the important variable. The vision projector remains the
official BF16 file in every profile.

The checked-in OpenCode configuration exposes the default as
`halo-ai/qwen3.8-27b`, alongside `qwen3.8df2`, `ds4`, `qwen3.8fp4`, and `qwen3.8fp8` under one
provider. These mutually exclusive managed LLM runtimes share the loopback
endpoint on port 8000, with text and image inputs enabled for DFlash2.
The Qwen3.8 entries default to `medium` reasoning and expose OpenCode
variants for `none`, `low`, `medium`, and `xhigh`. The thinking variants use
Qwen's recommended temperature 1.0 and top-p 0.95 sampling policy; `xhigh`
remains an explicit opt-in for unusually difficult requests.
`halo-ai test qwen3.8fp4` also performs a live template canary: an omitted
effort must render byte-for-byte like explicit `medium`, while the `xhigh`
control must differ, before the ordinary non-thinking smoke can pass.

The experimental `qwen3.8-27b-q6xl-vision-lemonade` profile reuses the same
on-disk Q6 XL target and BF16 projector through Lemonade's real ROCm/HIP
llama.cpp backend. It holds the target-only Vulkan profile's native 262K
context, F16 K/V, one slot, Flash Attention, and 4096/4096 batch settings fixed
for a backend A/B test. Halo can register a local target, projector, and draft
as one first-class Lemonade 11.8 model. The resulting
`qwen3.8-27b-q6xl-vision-dflash2-lemonade` profile is deliberately disabled,
however. Historical 11.7 stable `b10597`/build 10594 and nightly `b1315` tests
both reject the valid
58-tensor DFlash2 draft with `expected 81, got 58`. Lemonade's companion wiring
works, but the supported 11.8.1 server/backend combination must pass the same load and
equal-history canaries before this gate is removed. Compare target-only ROCm
directly with `qwen3.8-27b-q6xl-strix-vision` in the meantime.
The first controlled 81-token prompt / 256-token generation A/B measured median
decode at 8.05 tok/s on Lemonade HIP and 8.44 tok/s on Vulkan. HIP won the first
uncached prefill 138.84 to 65.10 tok/s. See
[`docs/results/qwen3.8-q6xl-vulkan-rocm-ab-2026-08-23.json`](../results/qwen3.8-q6xl-vulkan-rocm-ab-2026-08-23.json)
for the exact runtime identities and samples.
The deprecated Lemonade 11.7 qualification record repeated the vision canary in three fresh processes
and reran both target-only controls without prompt caching. ROCm median
prefill/decode was 130.07/8.013 tok/s versus Vulkan 99.69/8.466 tok/s. See
[`docs/results/qwen3.8-q6xl-lemonade-dflash2-compat-2026-08-23.json`](../results/qwen3.8-q6xl-lemonade-dflash2-compat-2026-08-23.json).

## Qwen3.8 Flash Next UD-Q4_K_XL and MTP

The experimental `qwen3.8-flash-next-ud-q4-k-xl-llamacpp` profile runs the
four-shard Unsloth quant through Halo's standalone ROCm llama.cpp image at 32K
context. Preview or re-check the exact artifact closure before starting it:

```bash
halo-ai profiles acquire qwen3.8-flash-next-ud-q4-k-xl-llamacpp --dry-run
halo-ai models verify qwen3.8-flash-next-ud-q4-k-xl --full
halo-ai start qwen3.8-flash-next-ud-q4-k-xl-llamacpp --switch
halo-ai test qwen3.8-flash-next-ud-q4-k-xl-llamacpp

# External-head MTP trial
halo-ai profiles acquire qwen3.8-flash-next-ud-q4-k-xl-mtp-llamacpp --dry-run
halo-ai profiles acquire qwen3.8-flash-next-ud-q4-k-xl-mtp-llamacpp
halo-ai start qwen3.8-flash-next-ud-q4-k-xl-mtp-llamacpp --switch
halo-ai test qwen3.8-flash-next-ud-q4-k-xl-mtp-llamacpp
```

The pinned four-shard set is 111,334,654,784 bytes (103.69 GiB). All four
local files were SHA-256 verified against Hugging Face revision
`38bb39ee97821de2c9009abb7e93950eec396e66` on 2026-09-02. Acquisition keeps
the repository's `UD-Q4_K_XL/` source subdirectory while preserving the
existing flat local directory. The optional cataloged shared Q4_K_M MTP head
adds 1,907,151,936 bytes (1.78 GiB), for a selected MTP closure of
113,241,806,720 bytes (105.46 GiB). Its locally verified SHA-256 is
`f521868a9e143718bef513772f6e04d9642551e362cf2439636d2abdbd149dfc`
at the same pinned revision. A separately selectable 2,786,568,256-byte shared
Q8_0 head is locally verified as
`5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6`;
profiles never mount or account for both heads at once.

No supplementary file is required for the target-only profile: the tokenizer
and chat template are embedded in shard 1. The MTP profile additionally mounts
the exact shared Q4_K_M head, passes it explicitly with `--spec-draft-model`,
uses `--spec-type draft-mtp --spec-draft-n-max 2`, and offloads the draft to the
GPU. Both profiles allow 1,200 seconds for a cold 105 GiB-class load; a 503 with
`Loading model` before then is readiness progress rather than a server failure.

Do not route this profile through Halo's qualified Lemonade configuration yet.
Lemonade Server 11.8.1 can register arbitrary GGUFs, but the pinned stable ROCm
package `b10597` contains llama.cpp build 10594, which predates upstream
`qwen4exp` support. Halo therefore builds
`localhost/halo-ai-llamacpp:b10715-mtp` from the immutable ROCm 10 gfx1151
base plus Unsloth's hash-pinned build 10715 archive at source commit
`92cedc8679d145902ead3f006258e8672eac11e6`. This is the first pinned release
here with the qwen4exp MTP graph and shared-target tensor borrowing. The
profiles enable Q8 K/V, mmap loading, one slot, explicit all-layer offload, the
published gfx1151 attention-rotation workaround, and hipBLASLt. Automatic fit
is disabled because all memory-sensitive settings are explicit; otherwise
build 10715 performs an unnecessary target-only sizing load and cannot count
the shared head correctly.

This is an `experimental-lockup` profile because its weights alone occupy most
of the 123.5 GiB CPU-visible memory. Start at 32K and monitor GTT, available
memory, swap, and kernel errors. Unsloth recommends the shared Q8_0 head for
maximum speed; the locally available shared Q4_K_M head is smaller but measured
about two acceptance points lower upstream. Keep `n=2` as the qualification
baseline. Benchmark `n=3` and `n=4` with greedy code and prose separately before
raising it; acceptance alone is not a speed result.

EngramHalo.cpp's `draft-mtp,ngram-mod`, `n=4`, `p-min=0.75` recipe is promising
for its own ROCm/HIP fork and Q8 head, but it also depends on Strix-specific QSA,
lazy-read, and kernel patches. The ordinary build-10715 profile therefore stays
at `n=2`; the isolated v0.7.5 candidates below makes the wider settings explicit A/B
candidates and requires decode throughput, acceptance, prompt depth, and exact
output equality at temperature zero before promotion.

### Advanced Strix Vulkan v0.7.5 candidates

Halo also carries `strixvulkan075` as an isolated candidate engine. It pins
Nathan's v0.7.5 amd64 image digest, build 10677, source commit
`dff60048744f99cb0af68d02be2314456b3269dc`, and bundled Mesa 26.3. Installing
it does not replace the build-10577 `strixvulkan` image or change any alias:

```bash
halo-ai install strixvulkan075

# Smaller Q6 control and best-guess general assistant profile.
halo-ai start qwen3.8-27b-q6xl-strix075-65k-vision-baseline --switch
halo-ai start qwen3.8-27b-q6xl-strix075-65k-vision-dflash2-advanced --switch
# Measured speed winner and lower-memory community alternative.
halo-ai start qwen3.8-27b-q6xl-strix075-65k-vision-dflash2-q8 --switch
halo-ai start qwen3.8-27b-q6xl-strix075-65k-vision-dflash2-iq4-xs --switch

# Existing 103.69 GiB Flash target: control, best guess, and width controls.
halo-ai start qwen3.8-flash-next-ud-q4-k-xl-strix075-baseline --switch
halo-ai start qwen3.8-flash-next-ud-q4-k-xl-strix075-mtp-advanced --switch
halo-ai start qwen3.8-flash-next-ud-q4-k-xl-strix075-mtp-n2 --switch
halo-ai start qwen3.8-flash-next-ud-q4-k-xl-strix075-mtp-n4 --switch
halo-ai start qwen3.8-flash-next-ud-q4-k-xl-strix075-mtp-q8-n2 --switch
halo-ai start qwen3.8-flash-next-ud-q4-k-xl-strix075-mtp-q8-n4 --switch
```

The first Q6 advanced guess uses the already-installed Q6 XL target, BF16 projector,
and Q4_K_M DFlash2 sidecar at 65K context, 2048/512 batch sizes, draft width 6,
probability floor 0.10, F16 target K/V, and Q8 draft K/V. A width-3 alternative
is cataloged separately. The Flash candidates reuse the existing four target
shards and a selectable Q4_K_M or Q8_0 shared head at 32K, with Q8 target/draft
K/V, widths 2--4, probability floor 0.75, mmap with lazy reads, no repacking,
no host buffers, and a medium reasoning effort capped at 2,048 tokens. See
[`docs/qwen3.8-flash-test-plan.md`](../qwen3.8-flash-test-plan.md).

The initial same-host Q6 screen selected width 6 and rejected ubatch 2048. At
4K+64 Q4 cut wall time 9.5% versus target-only; at 32K+64 it more than doubled
decode but was 9.5% slower overall because of speculative prefill overhead. Its
estimated 32K break-even is about 281 generated tokens. This is one repetition
per performance arm. The follow-up draft-weight matrix found Q8_0 fastest
end-to-end at both depths. IQ4_XS used about 1 GiB less peak GTT than Q8 but was
1.3--1.8% slower; BF16 was slower and heavier than both. Two fresh baseline,
Q4, and Q8 processes all scored 10/13 and matched all 13 output-token streams,
proving strict identity for Q4 and Q8 under the fixed suite. No new target model
was downloaded. `qwen3.8-27b` now selects the Q8 profile as the balanced Qwen3.8
default; the existing native-262K `qwen3.8df2` alias remains unchanged. Exact measurements are in
[`docs/results/qwen3.8-q6xl-strix075-advanced-2026-09-10.json`](../results/qwen3.8-q6xl-strix075-advanced-2026-09-10.json).

The Flash-Next screen selected Q4-head width 4 as a performance-only option,
not a default. It decoded 59.4% faster at 4K+64 and 87.0% faster at 32K+64,
but speculative prefill made the 32K short request 15.1% slower end-to-end. At
4K+1,024 it was 35.8% faster end-to-end; at 31,743+1,024 it was still 0.94%
slower, with a crossover beyond the remaining context budget. Two fresh Q4 and
two fresh Q8 quality processes were internally consistent but each matched
only 12/13 target token streams. Q8 width 4 was also 4.5--4.8% slower than Q4
by two-pass wall-time median and used about 0.8--0.9 GiB more GTT. Within the
Strix Vulkan lane, target-only remains the reference/fallback, Q4 width 4 is
for long generation after shallower prompts, and Q8 remains a verified diagnostic. See
[`docs/results/qwen3.8-flash-next-strix075-advanced-2026-09-10.json`](../results/qwen3.8-flash-next-strix075-advanced-2026-09-10.json).

## Qwen3.8 ROCmFP4 baseline

The Stage 1 Qwen3.8 profile is text-only and uses a dedicated q38rocm ROCmFPX
ABI over `Vulkan0`; it is intentionally separate from the ordinary llama.cpp
engine. Preview the exact acquisition closure before making any download:

```bash
halo-ai profiles acquire qwen3.8-27b-rocmfp4-baseline --dry-run
halo-ai profiles acquire qwen3.8-27b-rocmfp4-baseline
halo-ai models verify qwen3.8-27b-rocmfp4 --full
halo-ai start qwen3.8-27b-rocmfp4-baseline --switch
halo-ai test qwen3.8-27b-rocmfp4-baseline
halo-ai stop qwen3.8-27b-rocmfp4-baseline
```

The only model artifact selected by that profile is the pinned
`Qwen3.8-27B-ROCmFP4-FAST.gguf`: 14,562,236,384 bytes (13.56 GiB) under
`/srv/halo-ai/models/julianmb/Qwen-3.8-27B-ROCmFP4-FAST-GGUF`. The dry-run
explicitly records FP8, NPU, vision, BF16, and reference artifacts as excluded.
The runtime image is built from immutable base-image digests and the q38rocm
v1.0.0 archive after exact SHA-256 verification. The qualified image has an
apparent size of 3,877,844,308 bytes (3.61 GiB), although its base layers may
already exist in rootless Podman storage. The archive binary reports
short source revision `e87d53e`; upstream has not published a resolvable full
commit for that claim, so Halo records it as unresolved rather than presenting
the q38rocm release-tag commit as the engine source revision.

Rollback is non-destructive: `halo-ai stop qwen3.8-27b-rocmfp4-baseline` stops
the service while preserving the verified GGUF and image. Start the previous
profile with `--switch`; remove the local ROCmFPX image separately only when
its cached runtime is no longer wanted.

The same-host Stage 1 qualification, including short/4K/32K cold-prompt
measurements and the direct `llama-bench` control, is recorded in
[`docs/results/qwen3.8-rocmfp4-baseline-2026-08-16.json`](../results/qwen3.8-rocmfp4-baseline-2026-08-16.json).
Performance claims collected on other q38rocm setups are not used as pass/fail
evidence for this host; these measurements are the baseline for subsequent
optimization.

The enabled `qwen3.8-27b-rocmfp4-mtp-conservative-q5-draft` profile reuses the
same GGUF and downloads no model weights. It uses `n_max=2`, `p_min=0.85`, and
q5_1 draft K/V. Its bounded quality score matched the baseline at 9/13, but it
did not satisfy strict token identity, so it remains an explicit experimental
choice and `qwen3.8fp4` still resolves to the unassisted baseline. The faster
`n_max=6` profiles remain gated. Performance results and the conformance caveat
are in
[`docs/results/qwen3.8-rocmfp4-mtp-2026-08-16.json`](../results/qwen3.8-rocmfp4-mtp-2026-08-16.json).

The already-present, hash-verified ROCmFP8 artifact is exposed only through the
explicit `qwen3.8-27b-rocmfp8-baseline` and `qwen3.8-27b-rocmfp8-mtp` GPU
profiles. A matched single-run 4K/32K screen found no FP8 performance reason to
change defaults: FP8 baseline decode was about 35% slower than FP4, and FP8 MTP
was tied at 4K but 9.1% slower at 32K than FP4 MTP while using about 12 GiB more
GTT. Keep FP8 for a future fixed quality comparison. The aligned record is
[`docs/results/qwen3.8-fp4-fp8-exact-context-2026-08-17.json`](../results/qwen3.8-fp4-fp8-exact-context-2026-08-17.json).

Turn those records into a repeatable gate with:

```bash
halo-ai tune mtp-compare \
  docs/results/qwen3.8-rocmfp4-baseline-2026-08-16.json \
  docs/results/qwen3.8-rocmfp4-mtp-2026-08-16.json

# With the selected profile already active, issue requests inside its container.
halo-ai bench rocmfpx-context qwen3.8-27b-rocmfp4-mtp \
  --prompt-tokens 4095,31998 --completion-tokens 64 \
  --prompt-pattern unique --repetitions 3 --output RESULT.json
halo-ai tune context-compare BASELINE.json CANDIDATE.json
```

The MTP comparison accepts only matching prompt-token sets and the same model
SHA-256 and calculates TTFT/prefill/decode/GTT deltas. The conservative profile
is available under an explicit experimental quality policy; strict token
identity remains unresolved, so it is not the default alias. External NPU and
cross-version drafting remain deferred research and are not part of the active
FP4/FP8 GPU tuning loop.

Engine build 213 accepts `ngram-mod,draft-mtp` and the `24/48/64` ngram flags
syntactically, but refuses model load with strict-Qwen MTP: ngram-mod disables
recurrent rollback, while strict MTP requires rollback covering the full draft.
Halo therefore does not expose that non-starting combination as a profile.
