# Qwen3.8 Flash Next UD-Q4_K_XL qualification

Current preparation state (2026-09-10):

1. The four Unsloth `UD-Q4_K_XL` shards are present under `/srv`, total
   111,334,654,784 bytes, and their local SHA-256 values match Hugging Face
   revision `38bb39ee97821de2c9009abb7e93950eec396e66`.
2. The shared Q4_K_M MTP head is present at
   `/srv/halo-ai/models/unsloth/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`.
   Its exact size is 1,907,151,936 bytes and its locally calculated SHA-256 is
   `f521868a9e143718bef513772f6e04d9642551e362cf2439636d2abdbd149dfc`;
   both match the pinned Hugging Face LFS object. The combined base-and-draft
   closure is 113,241,806,720 bytes (105.46 GiB).
   The upstream shared Q8_0 head is independently cataloged at 2,786,568,256
   bytes with SHA-256
   `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6`;
   Q4 and Q8 profiles select only their own head.
3. The catalog provides separate target-only and external-head MTP profiles at
   32,768 context. Both use explicit all-layer offload, Q8 K/V, mmap, one slot,
   batch 8192, ubatch 2048, four decode threads, 32 batch threads, Flash
   Attention, hipBLASLt, disabled automatic fitting, and the gfx1151
   attention-rotation workaround.
4. Lemonade Server 11.8.1 stable remains unsuitable: its pinned ROCm backend is
   llama.cpp build 10594, before `qwen4exp` support. The old standalone build
   10711 loads the target but cannot borrow the shared MTP tensors.
5. Halo now derives `localhost/halo-ai-llamacpp:b10715-mtp` from the immutable
   Kyuz0 ROCm 10 gfx1151 base and Unsloth release
   `b10715-mix-86bd2d3`, source commit
   `92cedc8679d145902ead3f006258e8672eac11e6`. The 351,903,376-byte release
   archive is verified before extraction against SHA-256
   `e77e58a117db2c43b6435d7561aa7362288d2b7891b47e723136f90adc71d5d6`.
6. A separate `strixvulkan075` lane pins Nathan v0.7.5 build 10677 at source
   commit `dff60048744f99cb0af68d02be2314456b3269dc` and amd64 image digest
   `sha256:cdb88888ed5547e03c377ea0525f3f281bd6f159e77fc3faf810fe0f24d6a146`.
   The image passed its immutable label check and exposed `Vulkan0` as
   `RADV STRIX_HALO` on the target host. It is a candidate; the existing
   build-10577 Strix engine and `qwen3.8df2` alias remain unchanged; the
   separately qualified Q8 profile is exposed through the new `qwen3.8-27b`
   balanced-default alias.

## Initial MTP qualification

The MTP profile must render these options in addition to the target model:

```text
--spec-type draft-mtp
--spec-draft-model /models/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
--spec-draft-ngl all
--spec-draft-n-max 2
```

Automatic fitting is disabled because the profile already supplies explicit
context, batch, and offload settings. Leaving it enabled first loads the target
for a sizing pass and tries the shared head alone, which logs
`borrow_shared_tensor ... load it as a draft of its target model`; that sizing
pass also omits the draft's memory. A rendered Halo profile should therefore
contain `--fit off` and should proceed directly to the paired load.

Qualify `n=2` first. A healthy `/v1/models` response proves only that loading
finished. `halo-ai test` must also observe nonzero drafted and accepted tokens
in server timings. Run target-only and MTP at temperature zero with identical
prompts and record:

- cold-load time and peak GTT/RSS;
- prompt and decode tokens per second;
- drafted-token acceptance percentage and accepted tokens per generation;
- exact normalized output equality;
- kernel OOM/reset errors and shutdown behavior.

The profile has a 1,200-second startup allowance because the 105 GiB mmap load
can legitimately return HTTP 503 `Loading model` for several minutes.

## Tuning order

1. Keep the Q4_K_M head, 32K context, and `n=2`; establish a repeatable
   target-only versus MTP baseline with greedy code and prose workloads.
2. Trial `n=3`, then `n=4`, changing no other option. Retain a candidate only
   if end-to-end decode throughput improves without a quality regression;
   acceptance percentage by itself is not sufficient.
3. If storage and memory permit, acquire the upstream-recommended shared Q8_0
   head and repeat the same matrix. Upstream measured about 66.1% acceptance
   for Q8_0 versus 64.4% for Q4_K_M and describes Q8_0 as the fastest choice.
4. Evaluate EngramHalo.cpp separately. Its reported
   `--spec-type draft-mtp,ngram-mod --spec-draft-n-max 4 --spec-draft-p-min 0.75`
   settings depend on its Strix-specific QSA, lazy-read, and ROCm/HIP changes;
   they are not defaults for Unsloth build 10715 or the Q4 head.
5. Increase context only after the selected MTP configuration survives repeated
   32K load, generation, and clean-stop cycles.

## Disposition of the linked ideas

- The [Strix Halo lab stack](https://pwilkin.github.io/strix-halo/) uses an
  IQ4_XS target, IQ4_XS DFlash2, and a custom retained-PM4 ROCm runtime. Its
  width, probability-floor, draft-K/V, batch, and long-prompt findings informed
  Halo's controlled profiles, but its headline numbers are not directly
  comparable to the installed Q6 XL or Flash-Next targets.
- The agentionai `Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF` is a different
  main-model quantization. It is not a drop-in configuration change for the
  existing 103.69 GiB UD-Q4_K_XL target, so evaluating it remains outside this
  no-new-main-download pass.
- [`rocm_wmma_gemm`](https://github.com/adelj88/rocm_wmma_gemm) is a standalone
  HIP/WMMA GEMM library with shape-specific tuning files. The pinned llama.cpp
  runtimes do not expose it as a backend or launch flag, so copying its gfx1151
  tuning JSON into Halo would have no effect. It needs a separate runtime
  integration and end-to-end qualification before becoming a profile option.
- The community Flash-Next reports motivated the v0.7.5, MTP-width, and Q8-head
  arms. Local exact-prompt wall time, memory, acceptance, and fixed-quality
  identity—not reported headline throughput—determine promotion here.

The smaller Q6 family profiles use DFlash2. Flash-Next is a different
`qwen4exp` architecture and uses its external shared MTP head through
`--spec-type draft-mtp`; substituting the Q6 DFlash2 sidecar would be invalid.

## Advanced v0.7.5 matrix

The v0.7.5 matrix uses the same already-present target and Q4_K_M shared head.
No new main-model download is selected by any of these profiles:

| Profile | Purpose | Draft width |
| --- | --- | ---: |
| `qwen3.8-flash-next-ud-q4-k-xl-strix075-baseline` | Matched target-only control | none |
| `qwen3.8-flash-next-ud-q4-k-xl-strix075-mtp-n2` | Conservative MTP control | 2 |
| `qwen3.8-flash-next-ud-q4-k-xl-strix075-mtp-advanced` | Educated best guess | 3 |
| `qwen3.8-flash-next-ud-q4-k-xl-strix075-mtp-n4` | Throughput challenger | 4 |
| `qwen3.8-flash-next-ud-q4-k-xl-strix075-mtp-q8-n2` | Q8 head control | 2 |
| `qwen3.8-flash-next-ud-q4-k-xl-strix075-mtp-q8-n4` | Q8 head throughput challenger | 4 |

All four hold context at 32K, batch/ubatch at 2048/512, target K/V at Q8,
one slot, mmap, lazy tensor reads, disabled repacking, disabled host buffers,
and a 2,048-token medium reasoning budget. The MTP arms additionally hold the
probability floor at 0.75 and draft K/V at Q8. The runtime does not expose a
speculative-adaptive CLI switch, so the catalog does not pretend that it does.

Run the baseline and width-3 guess first, each in at least two fresh processes:

```bash
halo-ai install strixvulkan075
halo-ai profiles acquire qwen3.8-flash-next-ud-q4-k-xl-strix075-baseline --dry-run
halo-ai profiles acquire qwen3.8-flash-next-ud-q4-k-xl-strix075-mtp-advanced --dry-run
halo-ai start qwen3.8-flash-next-ud-q4-k-xl-strix075-baseline --switch
halo-ai test qwen3.8-flash-next-ud-q4-k-xl-strix075-baseline
halo-ai start qwen3.8-flash-next-ud-q4-k-xl-strix075-mtp-advanced --switch
halo-ai test qwen3.8-flash-next-ud-q4-k-xl-strix075-mtp-advanced
```

Only run widths 2 and 4 after both primary arms load and stop cleanly. Promote
the winner only if it improves wall time on code and prose separately, reports
nonzero drafted and accepted tokens, remains within the memory reserve, and
matches the baseline's normalized greedy outputs. A higher acceptance ratio is
not a promotion criterion by itself.

### Same-host Q4-head result

The v0.7.5 target-only, Q4-head width-2, width-3, and width-4 arms all loaded,
generated, and stopped cleanly. One exact-token cold-cache repetition at each
prompt depth produced:

| Arm | 4K+64 PP / decode tok/s | 4K wall | 32K+64 PP / decode tok/s | 32K wall | Peak GTT |
| --- | --- | ---: | --- | ---: | ---: |
| target only | 299.15 / 22.34 | 16.60 s | 272.51 / 20.95 | 120.49 s | 76.9 GiB |
| Q4 head, width 2 | 275.53 / 34.74 | 16.76 s | 233.91 / 33.84 | 138.72 s | 79.2 GiB |
| Q4 head, width 3 | 275.01 / 33.85 | 16.83 s | 232.02 / 36.40 | 139.70 s | 79.2 GiB |
| Q4 head, width 4 | 277.58 / 35.61 | 16.60 s | 233.50 / 39.17 | 138.71 s | 79.2 GiB |

Width 4 is the Q4 decode winner: 59.4% faster than target-only at 4K and 87.0%
faster at 32K. For a 64-token completion, however, its speculative-prefill
cost leaves 4K wall time effectively tied and 32K wall time 15.1% slower. The
estimated end-to-end break-even is about 64 output tokens at 4K and 883 at 32K.

Two fresh target-only processes and two fresh width-4 processes each scored
10/13 and were internally consistent on all 13 fixed-quality cases. Width 4
matched only 12/13 target output-token hashes: the `code-slice-semantics` case
returned normalized `e` under every tested MTP width (including a temporary
width-1 diagnostic), versus target-only `eul`. Both answers fail that case, so
the bounded score is unchanged, but strict identity is not proven. The Q4 MTP
arms therefore remain performance-only experimental profiles and are not a
default route.

The smaller Q6 control pair lets the same runtime policy be screened without a
105 GiB model load:

```bash
halo-ai start qwen3.8-27b-q6xl-strix075-65k-vision-baseline --switch
halo-ai start qwen3.8-27b-q6xl-strix075-65k-vision-dflash2-advanced --switch
# Low-width alternative:
halo-ai start qwen3.8-27b-q6xl-strix075-65k-vision-dflash2-n3 --switch
# Draft-weight controls:
halo-ai start qwen3.8-27b-q6xl-strix075-65k-vision-dflash2-q8 --switch
halo-ai start qwen3.8-27b-q6xl-strix075-65k-vision-dflash2-iq4-xs --switch
halo-ai start qwen3.8-27b-q6xl-strix075-65k-vision-dflash2-bf16 --switch
```

The Q6 advanced profile is a best guess rather than a copied headline: it uses
the installed Q6 XL target and Q4_K_M DFlash2 sidecar, 65K context,
2048/512 batch/ubatch, width 6,
probability floor 0.10, F16 target K/V, and Q8 draft K/V. This preserves the
current high-quality artifacts while applying the community findings that low
acceptance prose can favor a shorter draft and that draft-cache precision must
be tuned independently from the target cache.

The initial same-host screen selected width 6 as the advanced Q6 guess. Width 6
and width 3 were tied within 0.2% wall time at 4K+64; width 6 was 2.0% faster
end-to-end and 7.6% faster in decode at 32K+64. Ubatch 2048 lost to 512 at both
prompt depths. The 32K width-6 arm still needs roughly 281 generated tokens to
amortize its slower speculative prefill, so target-only remains preferable for
short, very long-prompt requests. The exact one-repetition screen and its
limitations are recorded in
`docs/results/qwen3.8-q6xl-strix075-advanced-2026-09-10.json`.

A matched draft-weight screen then selected Q8_0 for speed: it completed the
4K+64 and 32K+64 requests in 23.90 and 183.59 seconds. IQ4_XS used about 1 GiB
less peak GTT but was 1.3--1.8% slower, making it the memory-optimized option;
BF16 was slower and heavier. Two fresh processes for target-only, Q4_K_M, and
Q8_0 each scored 10/13 and were self-consistent on all 13 cases. Both drafters
matched all 13 target token streams, so their strict-identity gates are proven
for this suite. The published 1.04 GB IQ4_XS sidecar was downloaded and verified
independently; no new target model was downloaded.
`qwen3.8-27b` now resolves to this Q8 profile. The native-262K `qwen3.8df2`
route remains available separately.

Do not add a projector: this catalog entry is text-only, and shard 1 already
contains its tokenizer and chat template.
