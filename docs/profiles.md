# Choosing and acquiring profiles

[Documentation index](README.md) · [Project README](../README.md)

A profile selects a model, runtime, context size, and serving settings. An alias
is a short name for a catalog profile. Run `halo-ai profiles list` to inspect
the installed catalog and `halo-ai profiles show PROFILE` for its settings and
availability.

## Common choices

These aliases reflect the checked-in catalogs. After updating the checkout,
use `./reload.sh` to deploy them to an existing installation.

| Profile or alias | Purpose | Details |
| --- | --- | --- |
| `qwen3.8-fn` | Recommended Flash-Next: Halogen v2, MTP, vision, 262K context | [Halogen](halogen-flash.md) |
| `qwen3.8-halogen-npu` | Halogen v2 plus NPU embeddings and reranking | [NPU setup](halogen-flash.md#npu-and-host-constraints) |
| `qwen3.8-27b` | Balanced 27B default: Q6 XL, vision, Q8 DFlash2, 65K context | [Qwen runtimes](qwen-profiles.md) |
| `qwen3.8df2` | Q6 XL vision and DFlash2 at native 262K context | [Qwen runtimes](qwen-profiles.md) |
| `qwen3.8fp4` / `qwen3.8fp8` | Text-only ROCmFPX baselines | [FP4/FP8](qwen-profiles.md#qwen38-rocmfp4-baseline) |
| `ds4` | DeepSeek V4 Flash, DSpark, disk KV, 384K Think Max | [DS4](ds4.md) |
| `seamless-m4t-v2-large-speech` | Speech-to-speech translation | [Speech](speech.md) |

These names describe catalog choices, not universal quality or performance
rankings. High-context and speculative profiles retain experimental limits.
In particular, the recommended Halogen profile still has an experimental risk
label, and DS4's longer contexts have allocation/smoke evidence rather than
full-length prompt qualification. Read the relevant guide before loading one.

The installed default is configured by `HALO_AI_DEFAULT_PROFILE`; the checked-in
template uses `qwen3.6-35b-a3b-q8xl-lemonade`. The family recommendations above
do not change that setting. Specify a profile explicitly in commands.

## Choose and acquire a profile

Complete [host installation](installation.md) and resolve `halo-ai doctor`
failures first. This example uses the recommended Halogen Flash-Next alias:

```bash
halo-ai profiles show qwen3.8-fn
halo-ai profiles render qwen3.8-fn
halo-ai profiles acquire qwen3.8-fn --dry-run
```

`show` and `render` inspect the selection without loading a model. The acquisition
preview lists the selected artifacts, missing downloads, and runtime. Review
storage and host memory requirements before proceeding:

```bash
halo-ai profiles acquire qwen3.8-fn
halo-ai start qwen3.8-fn --switch
halo-ai test qwen3.8-fn
halo-ai status
```

Acquisition downloads and verifies missing artifacts and installs the selected
runtime. It reuses existing artifacts. Starting a profile does not download its
missing model weights; `--switch` stops a conflicting managed runtime first.
For this profile, connect clients to `http://127.0.0.1:8731/v1` using model ID
`halogen-qwen3.8-flash-next-v2`. See the [Halogen guide](halogen-flash.md) for
serving defaults and validation limits.

When finished:

```bash
halo-ai stop
```

For other profiles, use the same inspect/acquire/start sequence with their
profile ID and follow the runtime-specific guide. NPU profiles additionally
require the documented host setup. See [Operations](operations.md#connect-a-client)
for each runtime's endpoint.

## Catalog and verification

The checked-in catalogs are [Strix Halo](../config/models.d/strix-halo.json) and
[Halogen](../config/models.d/halogen.json). Installation deploys them under
`/etc/opt/halo-ai/models.d`.

`halo-ai models verify` checks cataloged local models without loading them.
`halo-ai models verify MODEL_ID --full` additionally hashes the selected files
and writes an inventory record; use a model ID from the catalog, not a profile
alias. Full verification can read many GiB from disk.
