# Qwen runtimes

[Reference manual](README.md) · [Profiles](profiles.md)

Use [Halogen](halogen-flash.md) for the recommended Flash-Next service. This
chapter covers Qwen GGUF and ROCmFPX profiles. Acquire each profile before
starting it; use `profiles show` and `profiles render` to inspect its exact
artifacts, runtime, settings, and readiness.

## 27B aliases, vision, DFlash2, and OpenCode

| Alias or profile | Runtime and purpose |
| --- | --- |
| `qwen3.8-27b` | Balanced 65K Q6 XL vision profile with Q8_0 DFlash2 on Strix Vulkan v0.7.5 |
| `qwen3.8df2` | Experimental native-262K Q6 XL vision profile with Q4_K_M DFlash2, `n-max=5`, on the separately pinned older Strix fork |
| `qwen3.8-27b-q6xl-strix-vision` | Native-262K target-only Vulkan rollback |
| `qwen3.8-27b-q6xl-vision-lemonade` | Target-only ROCm/HIP comparison using the same Q6 XL target and BF16 projector |
| `qwen3.8fp4` / `qwen3.8fp8` | Text-only ROCmFPX baselines |

```bash
halo-ai profiles acquire qwen3.8-27b --dry-run
halo-ai profiles acquire qwen3.8-27b
halo-ai start qwen3.8-27b --switch
halo-ai test qwen3.8-27b
```

The 65K Q8 DFlash2 selection passed the fixed 13-case output-token identity
screen against target-only in two fresh processes. The native-262K Q4 route
also matched its target-only control on the equal-history suite and a structured
image canary. These are bounded integration checks, not general proof of model
quality or speculation fidelity at every prompt length.

The checked-in OpenCode configuration exposes `halo-ai/qwen3.8-27b`,
`qwen3.8df2`, `ds4`, `qwen3.8fp4`, and `qwen3.8fp8` under one provider at
`http://127.0.0.1:8000/v1`. Only one of these managed runtimes runs at a time.
DFlash2 entries accept text and images. Qwen entries default to `medium`
reasoning and expose `none`, `low`, `medium`, and `xhigh` variants. Thinking
variants use temperature 1.0 and top-p 0.95; `xhigh` is opt-in.

`halo-ai test qwen3.8fp4` checks that omitted reasoning effort renders like
explicit `medium` and differs from `xhigh`, before the non-thinking smoke test.

### Lemonade compatibility

Lemonade 11.8.1 uses the qualified stable ROCm package `b10597` (binary build
10594). Qwen3.6 profiles cover text, MTP, vision, and selected larger contexts;
use the catalog to choose a supported combination. For these Qwen3.6 profiles,
vision and MTP are separate selections. See [Lemonade operations](operations.md#lemonade-and-standalone-llamacpp).

The Qwen3.8 target-only Lemonade profile uses native 262K context, F16 K/V, one
slot, Flash Attention, and batch/ubatch 4096/4096. The
`qwen3.8-27b-q6xl-vision-dflash2-lemonade` profile remains disabled. Earlier
stable and nightly backend trials rejected its valid 58-tensor draft with
`expected 81, got 58`. The supported server/backend pair must pass load and
equal-history canaries before that gate can be removed.

A [recorded target-only comparison](results/qwen3.8-q6xl-vulkan-rocm-ab-2026-08-23.json)
measured 8.05 tok/s median HIP decode versus 8.44 Vulkan, with faster first
uncached prefill on HIP. Those runtime-specific measurements do not qualify
DFlash2 through Lemonade.

## Qwen3.8 Flash Next UD-Q4_K_XL and MTP

These text-only experimental profiles serve the four-shard Unsloth GGUF through
standalone ROCm llama.cpp at 32K context:

```bash
halo-ai profiles acquire qwen3.8-flash-next-ud-q4-k-xl-llamacpp
halo-ai start qwen3.8-flash-next-ud-q4-k-xl-llamacpp --switch
halo-ai test qwen3.8-flash-next-ud-q4-k-xl-llamacpp

# External shared-head MTP variant:
halo-ai profiles acquire qwen3.8-flash-next-ud-q4-k-xl-mtp-llamacpp
halo-ai start qwen3.8-flash-next-ud-q4-k-xl-mtp-llamacpp --switch
halo-ai test qwen3.8-flash-next-ud-q4-k-xl-mtp-llamacpp
```

| Selected artifacts | Disk bytes |
| --- | ---: |
| Four target shards | 111,334,654,784 (103.69 GiB) |
| Shared Q4_K_M MTP head | 1,907,151,936 (1.78 GiB) |
| Target plus Q4 head | 113,241,806,720 (105.46 GiB) |
| Alternative shared Q8_0 head | 2,786,568,256 |

Artifacts are pinned to revision `38bb39ee97821de2c9009abb7e93950eec396e66`;
expected hashes are in the [manifest](halo-ai.md#expected-file-manifest). Profiles
select only one shared head. The tokenizer and chat template are in shard 1;
no projector is needed.

The runtime is `localhost/halo-ai-llamacpp:b10715-mtp`, built from an immutable
ROCm 10 gfx1151 base and hash-pinned Unsloth build 10715. This supports the
`qwen4exp` MTP graph and shared-target tensor borrowing; Lemonade's pinned
build 10594 does not. MTP uses `--spec-type draft-mtp`, the explicit shared
head, full GPU draft offload, and draft width 2.

Both profiles use Q8 K/V, mmap, one slot, all-layer offload, hipBLASLt, and the
gfx1151 attention-rotation workaround. Automatic fitting is disabled because
the settings are explicit and a target-only sizing load cannot account for the
shared head correctly. Cold startup allows 1,200 seconds; HTTP 503 `Loading
model` within that interval is readiness progress.

These profiles carry `experimental-lockup` risk because weights alone occupy
most of the reference host's CPU-visible RAM. Keep the 32K baseline and monitor
GTT, available RAM, swap, and kernel errors. Use the [qualification procedure](qwen3.8-flash-test-plan.md)
before increasing draft width or context.

### Strix Vulkan v0.7.5 profiles

`strixvulkan075` pins build 10677, source
`dff60048744f99cb0af68d02be2314456b3269dc`, and Mesa 26.3. It is separate from
the older `strixvulkan` engine used by `qwen3.8df2`.

| Family | Selection | Qualification boundary |
| --- | --- | --- |
| Q6 XL 65K + vision | Q8_0 DFlash2, exposed as `qwen3.8-27b` | Fastest end-to-end in the bounded draft-weight screen; 13/13 token streams matched target-only |
| Q6 XL 65K + vision | IQ4_XS DFlash2 | About 1 GiB less peak GTT than Q8, 1.3–1.8% slower in that screen |
| Flash-Next 32K | Target-only | Reference/fallback for this runtime |
| Flash-Next 32K | Q4 shared head, width 4 | Faster long generation after shallow prompts; only 12/13 target token streams matched |
| Flash-Next 32K | Q8 shared head | Diagnostic; slower and heavier than Q4 in the local screen |

Q6 uses F16 target K/V; Flash-Next uses Q8 target K/V. Both use batch/ubatch
2048/512 and one slot. Speculative prefill can outweigh decode gains on long
prompts with short answers, so compare total request time as well as tokens/s.

The [Q6 measurements](results/qwen3.8-q6xl-strix075-advanced-2026-09-10.json)
and [Flash-Next measurements](results/qwen3.8-flash-next-strix075-advanced-2026-09-10.json)
record exact settings and limits. The [qualification chapter](qwen3.8-flash-test-plan.md)
lists the Flash-Next profile matrix and promotion criteria.

## Qwen3.8 ROCmFP4 baseline

ROCmFPX is a dedicated q38rocm ABI over `Vulkan0`, separate from ordinary
llama.cpp. The FP4 baseline is text-only and selects the pinned
`Qwen3.8-27B-ROCmFP4-FAST.gguf`, 14,562,236,384 bytes (13.56 GiB), under
`/srv/halo-ai/models/julianmb/Qwen-3.8-27B-ROCmFP4-FAST-GGUF`.

```bash
halo-ai profiles acquire qwen3.8fp4 --dry-run
halo-ai profiles acquire qwen3.8fp4
halo-ai start qwen3.8fp4 --switch
halo-ai test qwen3.8fp4
```

The image uses immutable base digests and a SHA-256-verified q38rocm v1.0.0
archive. Its binary reports short revision `e87d53e`; the full source revision
is unresolved and is not inferred from the release-tag commit. FP8, vision,
NPU, BF16, and reference artifacts are excluded from FP4 acquisition.

The explicit `qwen3.8-27b-rocmfp4-mtp-conservative-q5-draft` profile reuses the
same GGUF with `n_max=2`, `p_min=0.85`, and q5_1 draft K/V. It matched the
baseline's bounded quality score but failed strict token identity, so
`qwen3.8fp4` remains unassisted. Faster `n_max=6` profiles remain gated.

FP8 is available through `qwen3.8fp8` and explicit FP8 MTP profiles. A matched
single-run 4K/32K screen found slower decode and about 12 GiB more GTT use than
FP4; it does not establish a quality ranking. NPU and cross-version drafting
are outside the active FP4/FP8 tuning path. The engine accepts ngram/MTP flags
syntactically but rejects their combined strict-Qwen model load, so that
combination is not exposed as a profile.

For an already active experimental MTP profile, measure exact prompt depths and
score matching baseline/candidate records with:

```bash
halo-ai bench rocmfpx-context qwen3.8-27b-rocmfp4-mtp-conservative-q5-draft \
  --prompt-tokens 4095,31998 --completion-tokens 64 \
  --prompt-pattern unique --repetitions 3 --output RESULT.json
halo-ai tune context-compare BASELINE.json CANDIDATE.json
halo-ai tune mtp-compare BASELINE.json CANDIDATE.json
```

The MTP comparison requires matching prompt-token sets and model SHA-256.
See the [FP4 baseline](results/qwen3.8-rocmfp4-baseline-2026-08-16.json),
[MTP record](results/qwen3.8-rocmfp4-mtp-2026-08-16.json), and
[FP4/FP8 screen](results/qwen3.8-fp4-fp8-exact-context-2026-08-17.json)
for the recorded test conditions.
