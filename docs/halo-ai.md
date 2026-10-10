# Architecture and configuration reference

[Reference manual](README.md) · [Project README](../README.md)

## Runtime architecture

The Bash entry point delegates to a Python standard-library application. Catalog
profiles select a model, engine, context, companion artifacts, and serving
settings. The lifecycle manager runs one large inference runtime at a time in
rootless Podman; Halogen's optional NPU services run alongside its GPU model.

Models remain outside containers under `/srv/halo-ai/models`. Containers receive
only the selected read-only model files and GPU devices. Caches and runtime state
persist separately. APIs bind to loopback; use SSH forwarding for remote clients.
See [Operations](operations.md#connect-a-client) for endpoints.

| Component | Responsibility |
| --- | --- |
| `lib/halo_ai/cli.py` | CLI, lifecycle, health checks, and benchmark orchestration |
| `lib/halo_ai/engines.py` | Engine registry, ports, container names, and policy hooks |
| `lib/halo_ai/engine_halogen.py` | Halogen profile validation, argument rendering, and health policy |
| `lib/halo_ai/profile_artifacts.py` | Shared selection of target, draft, vision, and NPU artifacts |
| `lib/halo_ai/artifact_store.py` | Locked, resumable downloads, size/hash verification, and atomic publication |
| `lib/halo_ai/halogen_npu.py` | Pinned NPU dependencies and host checks |
| `lib/halo_ai/host_profile.py` | Guarded Limine GPU/NPU and GTT configuration |
| `lib/halo_ai/longbench.py` | Pinned, resumable LongBench-v2 evaluation |
| `lib/halo_ai/speech/` | Dedicated speech image and translation service |

Acquisition, readiness checks, memory accounting, and container rendering share
one artifact selection. Disk size and resident-memory estimates are distinct:
demand-paged lookup tables need not be fully resident. Engine policy modules
validate supplied values; callers own host and network I/O.

## Configuration

Configuration is parsed as data, without shell evaluation or expansion. Defaults
are overlaid by `/etc/opt/halo-ai/config.env`, then
`${XDG_CONFIG_HOME:-~/.config}/halo-ai/config.env`. `halo-ai --config PATH ...`
selects a single configuration file instead of those two; built-in defaults
still apply. Relative catalog/preset directories in that explicit file resolve
against its directory. Missing catalog/preset directories fall back to the
checkout's templates.

The [configuration template](../config/halo-ai.env.example) lists all settings
and exact image pins. Model files, companions, contexts, and compatible engines
belong in the [catalogs](../config/models.d/), while request-level sampling and
reasoning fields belong in [presets](../config/request-presets.d/).

| Setting | Purpose |
| --- | --- |
| `HALO_AI_RUN_USER` | Designated non-root runtime operator |
| `HALO_AI_MODELS_ROOT` | External model storage boundary |
| `HALO_AI_CATALOG_DIR`, `HALO_AI_PRESET_DIR` | Catalog and request-preset directories |
| `HALO_AI_STATE_DIR`, `HALO_AI_INVENTORY_FILE` | Durable lifecycle state and full-verification inventory |
| `HALO_AI_CACHE_DIR` | Rebuildable runtime cache root |
| `HALO_AI_MODELS_REQUIRE_MOUNT`, `HALO_AI_MODELS_EXPECT_UUID` | Require and identify the model filesystem |
| `HALO_AI_DEFAULT_PROFILE` | Default selection for commands that permit an omitted profile; `start` takes an explicit profile |
| Engine image and port settings | Pinned runtime image and loopback endpoint |
| `HALOGEN_XRT_LIB_DIR` | Host XRT library directory for NPU profiles; `/usr/lib` by default |
| `DS4_KV_CACHE_ENABLED`, `DS4_KV_CACHE_DIR`, `DS4_KV_CACHE_MB` | DS4 disk-cache policy, location, and budget |
| `HALO_AI_GTT_TARGET_GIB`, `HALO_AI_GTT_AUTOTUNE` | Desired aperture and memory-failure response policy |
| `HALO_AI_GTT_CANDIDATES_GIB`, `HALO_AI_GTT_MAX_GIB`, `HALO_AI_OS_RESERVE_GIB` | Allowed GTT progression, ceiling, and OS reserve |

Use `profiles show/render` and `presets show/render` to inspect effective
selections before loading a model. Configuration cannot make an incompatible
engine or companion pairing valid. See [host configuration](host-configuration.md)
for GTT policy and [operations](operations.md#memory-failures-and-recovery)
for pending trials.

## Storage and ownership

| Path | Role |
| --- | --- |
| `/opt/halo-ai/releases`, `/opt/halo-ai/current` | Root-owned content-addressed releases and active symlink |
| `/usr/local/bin/halo-ai` | Link to the installed entry point |
| `/etc/opt/halo-ai` | Host configuration, catalogs, and request presets |
| `/var/opt/halo-ai/state` | Trials, inventory, benchmark results, and rollback records |
| `/var/cache/halo-ai` | Rebuildable caches |
| `/var/lib/halo-ai-installer` | Root-only installer journal |
| `/srv/halo-ai/models` | External artifacts, preserved through uninstall |
| `${XDG_RUNTIME_DIR}/halo-ai` | Boot-scoped hardware and runtime state |

Only mutable state/cache leaves belong to the operator. Static release files
remain root-owned. Rootless Podman keeps images and named volumes in its own
store; inspect its location with `podman info`. Moving that store affects the
operator's unrelated containers and is outside host installation.

The installer validates the model root's separation from application paths; it
does not copy, chmod, or chown the model collection. If models live on a separate
filesystem, configure the mount requirement and expected UUID so a missing
mount cannot silently redirect downloads onto the root disk.

## Catalog and verification contract

Catalogs pin model identity, repository revision, ordered files, roles, sizes,
hashes, and compatible engines. Profiles reference catalog model IDs; aliases
resolve to canonical profiles, which are recorded in runtime state and results.
Discovery does not override those decisions.

Scanning is read-only. AppleDouble, temporary, partial, and hidden metadata files
are not models. GGUF files need valid magic and metadata, complete ordered shard
sets, and exact checkpoint-compatible companions. A projector or DSpark file
cannot serve as a main model.

`models verify` checks structure and exact sizes. `models verify MODEL_ID --full`
streams SHA-256 over files and writes the inventory; it does not load a runtime.
Expected catalog hashes are not proof that a particular host's files have been
verified. NPU auxiliary files have their own [manifest](../lib/halo_ai/halogen_npu.py).

## Expected-file manifest

Each entry is `sha256 bytes relative-path`, relative to the configured model
root. These are the catalog's expected values. The source suite checks this
manifest against every cataloged model file; local verification is a separate
operator action.

```text
0f50e9626df98168e7c6e0cc264e2a92b5184dd885a175a06628d979b5edceeb 1523566720 peonist-ai/halogen-qwen3.8-flash-next/qwen38-flash-next-mtp.hgn
d62e0ae553fe88afd3833733d4a4c669f34d20fd8dfce4b9610525bed2134b10 897916416 peonist-ai/halogen-qwen3.8-flash-next/qwen38-flash-next-vision.hgn
9c116bbc01f77b7a15464c1a124eb3325b286089b8a2a6f2856c9b246a235bd6 124068083904 peonist-ai/halogen-qwen3.8-flash-next/qwen38-flash-next-w4b.hgn
71246c6ab3fc1de2cf06326f18e275fe9c2a18366d646ed3357d194c884fc687 66687678432 peonist-ai/halogen-qwen3.8-flash-next/qwen38-flash-next-v2.hgn
3450acada94e19aabad88bd49b45eb70304820949c2240fc178067b20ae5dffe 51200246144 peonist-ai/halogen-qwen3.8-flash-next/qwen38-flash-next-ngram.hgn
f49c8d14fa972c1db5115c714981e399585a6a98106a1a055d9e77026bb6de4c 2478095488 peonist-ai/halogen-qwen3.8-flash-next/qwen38-flash-next-w4b.overlay-speed.hgn
1cdfc3a9f988955bfe9a71bb808d393030abbf9f99d34ffa1ef93815a49b39ab 2572466560 peonist-ai/halogen-qwen3.8-flash-next/qwen38-flash-next-w4b.overlay.hgn
c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041 8952 peonist-ai/halogen-qwen3.8-flash-next/tokenizer/chat_template.jinja
e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e 202 peonist-ai/halogen-qwen3.8-flash-next/tokenizer/generation_config.json
a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d 3353259 peonist-ai/halogen-qwen3.8-flash-next/tokenizer/merges.txt
0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3 12809320 peonist-ai/halogen-qwen3.8-flash-next/tokenizer/tokenizer.json
b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27 17928 peonist-ai/halogen-qwen3.8-flash-next/tokenizer/tokenizer_config.json
ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003 6722759 peonist-ai/halogen-qwen3.8-flash-next/tokenizer/vocab.json
659e22fbd01c9e13ea37a57c8d9c41e0a8819dffa3473d3c5286ee44b2d3398f 97591747456 antirez/deepseek-v4-gguf/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf
7e319924541db3f7a163ed7e11d7532a70d48228ab59d36cb81e1d4511885360 5989114272 antirez/deepseek-v4-gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf
dec1cee704800267d9d836d5a61aefc33705be939bbb3058fa9006d98191576d 5257696 unsloth/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf
3064d3c4c1d6363e9f9ad88e90a3e2c5fb2d6f7ae16ca72135c3ce6a5c984da5 49910532416 unsloth/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00002-of-00004.gguf
2e9b2732eca7da8324f731653624a4f5c9846258926fd9f468cc703afb51a019 49257859456 unsloth/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00003-of-00004.gguf
4ca79d8e5107dd1b9bb57b176a7c09948837425dee49f0f1dfd6547a3769fea7 5034198464 unsloth/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00004-of-00004.gguf
2c7ac54b0b64a99df1f139a9f1371a00198265e1d6a614b77597d20a655a4249 10896057440 unsloth/DeepSeek-V4-Flash-0731-GGUF/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf
3d6ff16be3258f910eac4dcec7142edc7a7100d8400fe363035c8cfedc151164 35776484480 unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q8_K_XL.gguf
fdc443e974cad1f61c45af1cfd5580855855ddce0d6c14cc500a5714c486ac1d 1842940480 unsloth/Qwen3.6-27B-MTP-GGUF/mmproj-F32.gguf
2770c8c7b8a1ad536168ea51463f3cf1b813e5b4d31f49ea0bf1f628b4688d05 4240 unsloth/qwen3.6-thinking.jinja
63f41b4f55a044a0c173b403bf901b0027fca2a717f63833fe24784deeb6f614 4333 unsloth/qwen3.6-nonthinking.jinja
6c6b816537abad90b250a0972b345466028d861ddfe316d5f0de31ca6440f781 39099447584 unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf
4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082 10946624 unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9 49859583136 unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf
56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3 49376141504 unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf
753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a 12087983520 unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf
f521868a9e143718bef513772f6e04d9642551e362cf2439636d2abdbd149dfc 1907151936 unsloth/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6 2786568256 unsloth/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
fb89c78d2be91cdb68eaaaa45b1270710bf34aa721dc1f0b9e3aa7b98d2e1da9 14562236384 julianmb/Qwen-3.8-27B-ROCmFP4-FAST-GGUF/Qwen3.8-27B-ROCmFP4-FAST.gguf
0bf5bfc9f946090af2d41b388ccb4d627e916c7250517c36a0de37d6eaccfd8e 28193396704 julianmb/Qwen-3.8-27B-ROCmFP4-FAST-GGUF/Qwen3.8-27B-ROCmFP8.gguf
701d8fa9ed214ab21bfc130cd2a7df19ca89bbef7713e2dfb19f3c63696aa917 25299061664 unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K_XL.gguf
83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53 931146432 unsloth/Qwen3.8-27B-GGUF/mmproj-BF16.gguf
18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594 1143006752 incoai/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
7f1c9a31a6ed40044c69f6508b50fd63b87abd8e1fb7fe4290303df549153751 2056414752 incoai/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q8_0.gguf
26af33a15b21475d668e4ee55639beea49932e7360b1144c6282721bcd127c14 3860293152 incoai/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-BF16.gguf
11c7848014bd68040a42837b381bbefff5d0acc22cf20b6055a48d560c834445 1038313376 ilintar/qwen3.8-27b-gguf-strix-halo/Qwen3.8-27B-DFlash2-IQ4_XS.gguf
9ac8a85d4e97d27fad026a813d52a069680e4e0cae701ef145b204bd533251b2 2066 facebook/seamless-m4t-v2-large/added_tokens.json
4b2fa9d863cc3033adaf261e6c3e32ad90347ee2f199a349ba1c77d5e26a605f 2716 facebook/seamless-m4t-v2-large/config.json
febbbbac4f0b122473a0125165c9291add850956315e35a025e16474cc0da5f4 9906948 facebook/seamless-m4t-v2-large/generation_config.json
85cab984fbc111f8713827c440499453b9e66f262862866eeed8725c302ba2ac 4999163080 facebook/seamless-m4t-v2-large/model-00001-of-00002.safetensors
9536dc05892a6ca8410bcfea763dde422e2430c3cc1f3acd26c55182c8989017 4238114628 facebook/seamless-m4t-v2-large/model-00002-of-00002.safetensors
79d33aa045308d1a93b11952f2a3e6a647c107b688e5fdf1124c4601626dcdb2 210713 facebook/seamless-m4t-v2-large/model.safetensors.index.json
59294da7ee216cf80f8083180cdb0ace7efabbe35a8b9009c5cab341af2bd1c7 1776 facebook/seamless-m4t-v2-large/preprocessor_config.json
026a76827537db9f1348e4d5aaa127bb10a2f2ff633243f3a52d16be82d73f9d 5165809 facebook/seamless-m4t-v2-large/sentencepiece.bpe.model
36ba9ab56ea9a4d0182ae18aed1ff16a15bf91b8aa544722c686744d5972b522 2337 facebook/seamless-m4t-v2-large/special_tokens_map.json
9e7f2075dbc38dbe11d2414bfa4fb8e900022e87bbff4f74c97817e32a7ab493 368901 facebook/seamless-m4t-v2-large/spm_char_lang38_tc.model
026a76827537db9f1348e4d5aaa127bb10a2f2ff633243f3a52d16be82d73f9d 5165809 facebook/seamless-m4t-v2-large/tokenizer.model
0a184dd2f5b9ee02ddfa7fc2110b7e919c471f29e2e771d8c18b22b7758827c8 19667 facebook/seamless-m4t-v2-large/tokenizer_config.json
```
