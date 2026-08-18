# Halo AI on CachyOS

## Purpose

This document is the implementation and operations plan for running local AI
services on this AMD Ryzen AI Max+ 395 (Strix Halo, `gfx1151`) system. The first two services are:

1. [Lemonade Server](https://developer.amd.com/playbooks/lemonade-getting-started/)
   for general local inference and an OpenAI-compatible API.
2. [DwarfStar/ds4](https://developer.amd.com/playbooks/deepseek-v4-flash-ds4/)
   for DeepSeek V4 Flash inference.

Both services will run in rootless Podman containers with direct AMD GPU access.
No host ROCm installation is required. A catalog-backed standalone llama.cpp
runner will cover advanced MTP, vision, and DSpark experiments that Lemonade does
not expose or validate; vLLM remains a deferred engine for native Hugging Face
checkpoints. Only one large GPU inference runtime will run at a time so it has
predictable access to unified memory.

This document is both the implementation record and operations runbook. The
checkout now contains a tested lifecycle manager, JSON catalog and presets,
LongBench-v2 adapter, resumable installer, guarded uninstaller, and Limine host
profile manager. Real Qwen, vision, MTP, DeepSeek, ds4, and DSpark loads have
completed their documented capability canaries; the measured matrix is kept in
`README.md`. The manual phase procedures remain the acceptance reference for
future runtime, model, and context changes.

## Source priority

When commands or version recommendations disagree, use sources in this order:

1. The current [AMD AI Playbooks](https://developer.amd.com/playbooks/).
2. The current upstream project documentation linked by the relevant AMD playbook.
3. This document's machine-specific notes.
4. Community field reports, used as diagnostic leads and verified against the
   first three sources before adoption. Examples:

   - [DeepSeek V4 Flash on Strix Halo](https://www.reddit.com/r/StrixHalo/comments/1v910m2/deepseek_v4_flash_up_to_32_toks_on_strix_halo/)
   - [DeepSeek V4 Flash 0731 discussion](https://www.reddit.com/r/StrixHalo/comments/1vir4c9/has_deepseek_v4_flash_0731_changed_the_game_for/)
   - [DSpark with DeepSeek](https://www.reddit.com/r/StrixHalo/comments/1vhi0gk/dspark_with_deepseek/)
   - [ds4 Flash 0731 field report](https://www.reddit.com/r/StrixHalo/comments/1vec4gy/ds4_flash_0731_amazing/)

5. Older blog posts, conversations, and cached instructions.

Before changing image versions, model names, kernel settings, or engine
arguments, recheck the relevant AMD playbook and upstream documentation. These
projects are moving quickly.

## Verified reference host

The following state was observed on 2026-08-09:

| Component | Observed value |
| --- | --- |
| System | GPD Win 5 |
| Processor | AMD Ryzen AI Max+ 395 |
| GPU | Radeon 8060S, Strix Halo, PCI ID `1002:1586` |
| Host OS | CachyOS (Arch-based) |
| Kernel | `7.1.6-1-cachyos-deckify` |
| Kernel GPU driver | In-tree `amdgpu` |
| Boot loader | Limine |
| Container runtime | Rootless Podman 6.0.2 |
| Physical unified memory | 128 GiB |
| GPU fixed VRAM pool reported by amdgpu | 2 GiB after BIOS correction (previously 96 GiB) |
| GTT aperture reported by amdgpu | 118 GiB (`126701535232` bytes) |
| CPU-visible RAM reported by Linux | About 123.5 GiB after BIOS correction |
| GPU device nodes | `/dev/kfd`, `/dev/dri/renderD128`, `/dev/dri/card1` |
| Vulkan | Mesa RADV, Vulkan 1.4 |
| NPU PCI function | Present as AMD device `1022:17f0` |
| NPU runtime state | Inactive: no bound `amdxdna` driver and no `/dev/accel` |

### Validated runtime checkpoints

The following end-to-end observations were recorded on 2026-08-09 after the
firmware memory correction. They are smoke-test evidence, not a formal
throughput benchmark:

| Profile | Approx. GTT used | Short-request observation |
| --- | ---: | --- |
| Qwen 27B baseline, 32K | 35.5 GB | Both reasoning modes passed; process contained `--spec-type none` and no projector |
| Qwen 35B-A3B baseline, 32K | 38.8 GB | About 48 tok/s non-thinking and 44 tok/s on one short thinking request |
| Qwen 27B MTP, 32K | 36.5 GB | 6/6 and 8/8 draft tokens accepted; about 12–14 tok/s |
| Qwen 35B-A3B MTP, 32K | 39.9 GB | 6/6 and 136/147 draft tokens accepted; about 55–58 tok/s on the repeated run |
| Qwen 35B-A3B baseline, 65K | 39.4 GB | Smoke request passed with `--ctx-size 65536` |
| Qwen 35B-A3B baseline, 128K | 40.8 GB | Smoke request passed with `--ctx-size 131072` |
| Qwen 35B-A3B MTP, 128K | 42.2 GB | Smoke request passed with nonzero accepted draft tokens |
| Qwen 27B baseline, 128K | 42.0 GB | Text-only smoke passed with no implicit projector |
| Qwen 27B vision, 128K | 44.1 GB | Generated-image canary answered `red` with the explicit projector |
| Qwen 27B standalone vision, 32K | 37.8 GB | Image canary passed on llama.cpp `b10335-74ce15741` |
| Qwen 27B standalone MTP, 32K | 36.7 GB | Draft acceptance passed; fixed sample 191.0 PP / 11.33 TPS |
| Qwen 35B-A3B standalone MTP, 32K | 40.1 GB | Draft acceptance passed; fixed sample 518.5 PP / 56.16 TPS |
| DeepSeek IQ3_XXS standalone, 32K | 104.5 GB | Fixed sample 56.78 PP / 10.21 TPS; about 19.1 GiB CPU memory available |
| DeepSeek IQ3_XXS + DSpark, 32K | 115.5 GB | Fixed sample 36.43 PP / 15.31 TPS; 24/27 drafts accepted; about 9.2 GiB available |
| DeepSeek V4 Flash ds4 hybrid, 32K | 115.5 GB | Smoke passed; only about 10.7 GB CPU memory remained available |

The initial four 32K Qwen processes used Lemonade ROCm package `b10334`, active llama.cpp binary
`b10333`, API fingerprint `b10333-08659901c`, one slot, F16 KV, and the pinned
non-thinking-default Jinja file. The request-level template canary proved both
thinking modes. Switching unloaded the old backend to about 142 MB observed GTT
before restarting the supervisor; the rootless container ID remained unchanged.
No kernel OOM event was observed in this corrected-topology boot.

Do not treat the short prompts above as a ranking by themselves. Use a fixed,
longer coding corpus with several repetitions before promoting an MTP profile as
the default. The evidence is sufficient to keep MTP available as the preferred
candidate for that benchmark.

The original firmware split reserved 96 GiB as fixed VRAM, so Linux owned only
about 32 GiB as ordinary RAM even though amdgpu advertised a 112 GiB GTT
aperture. That aperture was an addressable ceiling, not an additional 112 GiB
of backing memory. A Qwen 27B load exhausted the small CPU-visible pool and
triggered the global OOM killer.

After changing the BIOS dedicated/UMA allocation to the small dual-boot setting,
the verified topology is approximately 123 GiB CPU-visible RAM, 2 GiB fixed
VRAM, and a 118 GiB GTT aperture. This supplies real backing memory for dynamic
GPU allocations while retaining host headroom.

Current kernel command line:

```text
quiet nowatchdog splash rw rootflags=subvol=/@ \
root=UUID=79b59d5a-2791-4c9d-9622-4dfb4886ed43 \
amd_iommu=off amdgpu.gttsize=120832 ttm.pages_limit=30932992
```

`120832 MiB / 1024 = 118 GiB`, and `30932992 * 4096 bytes = 118 GiB`.
The earlier 112 GiB aperture already satisfied the ds4 playbook's recommendation
for at least a 110 GiB shared GPU-memory pool, but only when paired with the
corrected CPU-visible backing. `halo-ai doctor` therefore validates both values
and must fail a 96/32 fixed split even when `mem_info_gtt_total` looks large.

The NPU is visible on PCI but is not currently usable. The host boots with
`amd_iommu=off`, the kernel has not bound `amdxdna`, and `/dev/accel/accel0` does
not exist. This is expected for the present GPU-optimized profile and is not a
blocker for Lemonade ROCm or ds4.

## Target architecture

```text
Local applications
    |
    +-- halo-ai profile manager
           |
           +-- Lemonade              127.0.0.1:13305/api/v1
           +-- standalone llama.cpp  127.0.0.1:8080/v1
           +-- ds4                    127.0.0.1:8000/v1
           +-- Seamless speech        127.0.0.1:7860
           +-- vLLM (deferred)        127.0.0.1:8001/v1
                    |
                    +-- one rootless Podman container at a time
                    +-- /dev/kfd and /dev/dri
                    +-- minimal, read-only model mount selected by catalog
                    +-- persistent caches only where the engine requires them
```

The APIs bind to host loopback only. Remote clients should use SSH port
forwarding rather than exposing unauthenticated inference ports on the LAN.

Existing `llama-rocm-7.14` and `llama-vulkan-radv` Toolbx containers are useful
for interactive testing and benchmarks, but they are not dependencies of this
project and will not be modified.

## Current project layout

```text
~/Projects/source/ai/halo-ai/
├── README.md
├── docs/
│   └── halo-ai.md
├── install.sh                    # resumable guarded system installer
├── uninstall.sh                  # guarded installed-solution removal
├── bin/
│   └── halo-ai                   # Bash entry point
├── lib/halo_ai/
│   ├── cli.py                    # lifecycle/catalog/runtime implementation
│   ├── longbench.py              # pinned, resumable LongBench-v2 adapter
│   ├── rocmfpx_tune.py           # MTP and exact-prefix Stage 3 scoring gates
│   ├── host_profile.py           # guarded Limine profile implementation
│   └── speech/
│       ├── Containerfile         # pinned gfx1151 ROCm speech image
│       └── speech_server.py      # loopback Gradio UI and translation API
├── config/
│   ├── halo-ai.env.example       # non-secret defaults
│   ├── models.d/                 # model/engine-profile catalog
│   │   └── *.json
│   └── request-presets.d/        # planner/implementer sampling presets
│   │   └── *.json
└── tests/
    ├── container.sh              # read-only, network-disabled Podman test entry
    ├── test_cli.py               # synthetic safety and catalog tests
    └── smoke.sh                  # non-mutating checkout smoke suite
```

For checkout-only prototyping, runtime data should not be stored in the source
tree:

```text
~/.local/share/halo-ai/          # state owned by this project
~/.local/state/halo-ai/          # image pins, OOM history, tuning decisions
~/.cache/halo-ai/ds4-kv/        # optional ds4 KV cache
<external model root>/           # preserve provider/repository layout
├── antirez/<repository>/
└── unsloth/<repository>/
```

The system-integrated layout below supersedes these per-user defaults when the
project is installed under `/opt`.

### Current implementation boundary

Implemented and exercised end to end:

- Safe configuration parsing, catalog/preset validation, read-only scanning,
  exact-size/GGUF verification, and optional full SHA-256 inventory.
- Hardware/VRAM/GTT observation, rootless Podman checks, profile rendering,
  managed-label lifecycle operations, loopback endpoints, and external-model
  mount protection.
- Durable pre-load trial records, same-boot retry refusal, suspected-lockup and
  caught-OOM classification, and conservative future-boot pressure reductions.
- Resumable content-addressed host installation, guarded uninstallation,
  Snapper integration, and checksum-backed Limine GPU/NPU profile staging.
- Real Lemonade and standalone llama.cpp Qwen text, MTP, vision, 65K, and 128K
  starts; real ds4 hybrid/disk-cache and standalone DeepSeek IQ3_XXS/DSpark
  starts; a dedicated Seamless ROCm speech service; exact API/audio smoke tests;
  PP/TPS telemetry; and bounded LongBench-v2 samples.

Current boundary and deliberate deferrals:

| Item | State | Reason or next prerequisite |
| --- | --- | --- |
| 10 Lemonade/Qwen profiles | Enabled and tested | Includes MTP, vision, 65K, and 128K |
| 3 standalone Qwen profiles | Enabled and tested | Current llama.cpp image; build and digest recorded per trial |
| Standalone DeepSeek baseline and DSpark | Enabled and tested | High-memory profiles; one runtime at a time |
| ds4 hybrid and disk-KV variant | Enabled and tested | 32K remains conservative; cache restore passed across container recreation |
| ds4 IQ2_XXS / 126K | External prerequisite | Recommended smaller model is not installed |
| vLLM | Deliberately deferred | Installed inventory is GGUF; vLLM GGUF support remains experimental |
| NPU runtime | External prerequisite | `amdxdna` is not bound and `/dev/accel/accel0` is absent |
| Seamless/STS speech | Enabled and tested | Pinned Seamless M4T v2 Large safetensors service with loopback API/UI; separate from LLM containers |

`profiles list/show` is the authoritative readiness view. A future gate is a
safety state, not a missing-model fallback; gated profiles may still be rendered
for inspection without loading weights.

Do not reorganize or copy the provider/repository directories merely to suit an
engine. The halo-ai catalog selects exact main files, shard sets, companions,
compatible runtimes, and initial settings. Lemonade receives an exact two-file
allowlist containing the approved Qwen text mains so they can be switched
without container recreation; other engines receive only the selected profile's
files. The complete model root is never a runtime discovery source.
Lemonade-managed downloads and backend artifacts
remain in named Podman volumes, while externally supplied GGUFs are referenced
in place and consume no duplicate model storage.

### Recommended system-integrated layout

The source checkout is a development input, not the installed application or a
data directory. The installer uses this layout for the single designated
rootless operator:

```text
/opt/halo-ai/                         # root-owned, static installed files
├── releases/
│   └── <version-or-git-id>/
│       ├── bin/halo-ai
│       ├── lib/halo_ai/
│       ├── config/                       # release templates
│       ├── docs/halo-ai.md
│       ├── install.sh
│       └── uninstall.sh
└── current -> releases/<version-or-git-id>

/usr/local/bin/halo-ai -> /opt/halo-ai/current/bin/halo-ai
/etc/opt/halo-ai/config.env           # root:<operator-group> 0640 host configuration
/etc/opt/halo-ai/models.d/             # root-owned model/profile catalog
/etc/opt/halo-ai/request-presets.d/    # root-owned sampling/reasoning presets
/var/opt/halo-ai/state/               # small durable state and trial history
/var/cache/halo-ai/ds4-kv/            # disposable/rebuildable KV cache
/var/cache/halo-ai/speech/input1.wav  # hash-pinned AMD smoke input
/var/lib/halo-ai-installer/            # root-only resumable install journal
/srv/halo-ai/models/
├── antirez/
│   └── deepseek-v4-gguf/
├── facebook/
│   └── seamless-m4t-v2-large/        # pinned Transformers safetensors subset
└── unsloth/
    ├── DeepSeek-V4-Flash-0731-GGUF/
    ├── Qwen3.6-27B-MTP-GGUF/
    └── Qwen3.6-35B-A3B-MTP-GGUF/
/run/user/<uid>/halo-ai/              # boot-scoped rootless runtime files
```

This follows the Filesystem Hierarchy Standard: `/opt/halo-ai` holds static
add-on software, `/etc/opt/halo-ai` holds its host-specific configuration, and
`/var/opt/halo-ai` holds mutable package state. `/srv/halo-ai/models` is a good
fit for site-specific data served by the inference services. `/run/user/<uid>`
is ephemeral state for the running operator. `/opt/models` is therefore not the
preferred location: models are mutable site data, not installed application
files.

Use versioned, immutable release directories and switch `current` atomically.
This makes an application rollback a narrow symlink operation; a whole-root
Snapper rollback remains a recovery tool rather than the normal versioning
mechanism. Do not install `.git`, working-tree build artifacts, model files, or
mutable caches under `/opt/halo-ai`.

The installer must accept an explicit `HALO_AI_RUN_USER`, resolve it through the
account database, and refuse root. It creates root-owned parents and only grants
that operator access to mutable leaves. Static release files remain `root:root`
and non-writable by the operator. Model mounts are always read-only. Because a
rootless container may use a remapped UID, the final host access policy should
use readable files (commonly `0644`), a deliberate ACL, or a verified `keep-id`
mapping rather than assuming host ownership alone is sufficient. `doctor` must
warn that the current GGUFs are executable (`0755`) and world-readable, but it
must not change their modes automatically. Never compensate for an ownership
error with `0777`, a privileged container, or a writable model mount.

If the large model collection lives on another filesystem or device, mount or
bind it at the stable `/srv/halo-ai/models` boundary instead of changing every
container definition. Use a persistent filesystem UUID in `/etc/fstab`, keep the
mountpoint itself root-owned, and have `doctor`/`start` verify the configured
source with `findmnt -T /srv/halo-ai/models` before reading or downloading. If an
expected model mount is absent, stop; otherwise a downloader could silently fill
the underlying root disk. Mounting the collection is an explicit host operation,
never something `start` performs automatically.

Rootless Podman currently stores its images, containers, and named volumes in
`/home/michael/.local/share/containers/storage`; this was confirmed with
`podman info`. That is outside the source tree and is a functional initial
default, but it does not satisfy a strict requirement to keep runtime data out
of `/home`.
This host also has a Snapper config for `/home`; inspect its creation policy
before pulling large images or allowing Lemonade to populate large named
volumes, because snapshots of `@home` can retain changed container-store data.
Changing the rootless Podman `graphroot` is a user-wide storage migration that
can affect unrelated containers, not an ordinary halo-ai install step. If the
eventual requirement is *no persistent halo-ai bytes under `/home`*, first
replace project named volumes with explicit project bind mounts, or design a
dedicated Podman store/operator after inventorying and stopping all of that
user's containers. Do not silently rewrite `~/.config/containers/storage.conf`
from this project.

## Verified model inventory

The GGUF model tree was surveyed read-only on 2026-08-09. It contained 289,414,525,472
bytes (269.54 GiB) of real model data in nine valid GGUF files. Their exact byte
sizes match the current objects published by the corresponding Hugging Face
repositories. Six additional 4 KiB `._*.gguf` files are AppleDouble metadata
sidecars, not models. There are no symlinks, partial files, sparse files, or
hard-linked duplicates. `/srv` is the Btrfs `@srv` subvolume and currently has
about 1.2 TiB available.

The survey did not compute every GGUF SHA-256 digest. Therefore their current
integrity status is **structure and exact-size verified**, not fully hash
verified. The separately downloaded Seamless subset adds 9,258,124,450 bytes
(8.62 GiB); all 12 of those files were locally SHA-256 verified on 2026-08-09.
The lifecycle tooling must preserve that distinction per model.

| Model ID | Installed components | Real size | GGUF architecture | Baseline runtime | Initial context |
| --- | --- | ---: | --- | --- | ---: |
| `deepseek-v4-flash-ds4-hybrid` | Antirez hybrid `fixed-0731`; optional exact 0731 DSpark support | 90.89 GiB main; 96.47 GiB with companion | `deepseek4` | ds4 | 32,768 control; 16,384 first DSpark screen |
| `deepseek-v4-flash-0731-iq3xxs` | Unsloth IQ3_XXS shards 1–4; optional DSpark companion | 97.05 GiB main; 107.20 GiB with companion | `deepseek4` + `dflash` | standalone llama.cpp; Lemonade base only | 32,768 |
| `qwen3.6-27b-q8xl` | Q8_K_XL main; installed F32 vision projector | 35.04 GiB total | `qwen35` + `clip` | Lemonade or standalone llama.cpp | 32,768 |
| `qwen3.6-35b-a3b-q8xl` | Q8_K_XL main; no local vision projector | 36.41 GiB | `qwen35moe` | Lemonade or standalone llama.cpp | 32,768 |
| `qwen3.8-27b-rocmfp4` | ROCmFP4-FAST language/MTP GGUF; text only | 13.56 GiB | `qwen35` | dedicated q38rocm ROCmFPX/Vulkan | 32,768 |
| `qwen3.8-27b-rocmfp8` | Q8_0_ROCMFPX language/MTP GGUF; text only | 26.26 GiB | `qwen35` | dedicated q38rocm ROCmFPX/Vulkan | 32,768 |
| `seamless-m4t-v2-large` | Two safetensors shards plus processor/tokenizer files | 8.62 GiB | `seamless_m4t_v2` | dedicated ROCm speech service | N/A |

The sharded DeepSeek first file is intentionally only 5,257,696 bytes: it holds
GGUF metadata and zero tensors. The remaining three shards contain the model's
1,328 tensors. A size heuristic must not reject shard 1, and a launcher must not
accept an incomplete shard set.

### Expected-file manifest

The following SHA-256 values are the **published expected hashes** observed from
the upstream repositories on 2026-08-09. They become locally verified only after
`halo-ai models verify --full` hashes the installed bytes and records the result.
Each manifest entry is `sha256 bytes relative-path`:

```text
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
fb89c78d2be91cdb68eaaaa45b1270710bf34aa721dc1f0b9e3aa7b98d2e1da9 14562236384 julianmb/Qwen-3.8-27B-ROCmFP4-FAST-GGUF/Qwen3.8-27B-ROCmFP4-FAST.gguf
0bf5bfc9f946090af2d41b388ccb4d627e916c7250517c36a0de37d6eaccfd8e 28193396704 julianmb/Qwen-3.8-27B-ROCmFP4-FAST-GGUF/Qwen3.8-27B-ROCmFP8.gguf
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

### Catalog and scanner rules

The checked-in `config/models.d/*.json` files are templates; installed copies
under `/etc/opt/halo-ai/models.d` are the execution source of truth. Each model
entry must declare a stable ID, repository, exact main path or ordered shards,
byte count and expected digest per file, GGUF architecture, optional companions
and their roles, permitted engines, and safe initial settings. Execution
profiles reference a model ID plus one engine and its arguments. Local operator
catalog entries override checked-in templates, but discovery never overrides a
catalog decision.

`halo-ai models scan` is read-only and must:

- Stay on the configured model filesystem (`find -xdev`) and never follow
  symlinks.
- Ignore `._*`, temporary, partial, and hidden metadata files even when their
  suffix is `.gguf`.
- Require GGUF magic and parseable metadata; a filename is insufficient.
- Identify numbered shards as one ordered model and require every declared
  shard before enabling a profile.
- Classify `mmproj` and `dspark` files as companions, never independent main
  models, and bind them only to the exact catalog model/checkpoint.
- Report unexpected size, digest, path, owner, mode, and repository contents
  without deleting, renaming, changing permissions, or downloading anything.

`models verify` performs fast structure and exact-size checks. `models verify
--full` additionally streams SHA-256 over every declared file, records the
verification time and result in `/var/opt/halo-ai/state/model-inventory.json`,
and does not load the model into RAM. The scanner should warn about the six
AppleDouble sidecars and the current `0755` file modes; cleanup and permission
changes remain explicit operator actions.

### Qwen3.8 ROCmFP4 Stage 1 qualification

The `qwen38-27b-rocmfp4-baseline` profile was qualified on this gfx1151 host on
2026-08-16. It uses only `Vulkan0`, mounts the single verified FP4 GGUF
read-only, forces `--spec-type none`, and disables template thinking in the
deterministic smoke request. Acquisition reported zero additional bytes on the
preinstalled setup and named FP8, NPU, vision, BF16, and reference artifacts as
excluded. No optional model artifact was downloaded.

| Prompt tokens | TTFT | Prompt tok/s | Decode tok/s | Peak GTT | Peak fixed VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.43 s | 162.12 | 12.83 | 14.04 GiB | 1.84 GiB |
| 4,095 | 22.24 s | 184.45 | 11.90 | 14.03 GiB | 1.84 GiB |
| 31,998 | 260.48 s | 122.87 | 10.74 | 14.17 GiB | 1.89 GiB |

These are single cold-prompt, cache-disabled baseline measurements with
temperature 0, seed 1, no speculation, and up to 64 completion tokens. The
direct archive `llama-bench` control measured 12.15 ± 0.04 raw decode tok/s
over three 128-token repetitions. Values reported from other q38rocm setups are
not acceptance evidence for this host; optimization work should compare
against this same-host record. The complete machine-readable inputs, software
revisions, timings, and memory values are in
[`results/qwen38-rocmfp4-baseline-2026-08-16.json`](results/qwen38-rocmfp4-baseline-2026-08-16.json).

The runtime distribution is immutable at the archive and image-input level,
but one upstream provenance item remains open: the v1.0.0 binary reports build
213 and short revision `e87d53e`, for which q38rocm does not publish a
resolvable full source commit. Halo records both the q38rocm release commit and
this unresolved engine claim without conflating them.

### Qwen3.8 GPU-MTP experiment

`qwen38-27b-rocmfp4-mtp` reuses the MTP tensors in the same FP4 GGUF, so its
profile-scoped acquisition adds zero model bytes. It enables the fork's
boundary-safe strict-Qwen verifier with one slot and currently uses
`n_max=6`, `p_min=0.60`. The profile remains explicitly experimental.

| Prompt tokens | Baseline decode | MTP decode | Change | MTP TTFT | Acceptance | Peak GTT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 12.83 tok/s | 16.61 tok/s | +29.5% | 0.53 s | 15/19 | 17.20 GiB |
| 4,095 | 11.90 tok/s | 18.93 tok/s | +59.1% | 23.58 s | 42/49 | 19.04 GiB |
| 31,998 | 10.74 tok/s | 15.67 tok/s | +45.9% | 301.77 s | 43/43 | 19.06 GiB |

The gain is generation-side. At 32K, MTP reduced prompt throughput by 13.7%,
increased TTFT by 15.9%, and used about 4.90 GiB more peak GTT than baseline.
It is therefore useful for generation-heavy work but not the default for a
cold long prompt with a short answer.

The five-case conformance probe produced four stable exact token matches and
five semantically equivalent answers. Full cross-process identity is not yet
proven: even the unassisted Vulkan profile selected different compact,
pretty-printed, or fenced forms across clean processes despite temperature 0
and fixed request/server seeds. This keeps the Stage 2 correctness gate open.
Detailed timings, draft acceptance/run-position statistics, memory, and the
conformance note are recorded in
[`results/qwen38-rocmfp4-mtp-2026-08-16.json`](results/qwen38-rocmfp4-mtp-2026-08-16.json).

### Aligned FP4/FP8 context screen

The already-present `Qwen3.8-27B-ROCmFP8.gguf` matches its exact
28,193,396,704-byte size and SHA-256 and passes the cataloged 866-tensor GGUF
inventory. It is selectable only through the explicit GPU-only
`qwen38-27b-rocmfp8-baseline` and `qwen38-27b-rocmfp8-mtp` profiles; their
acquisition plans report zero additional bytes on this host.

The first exact-token screen used identical cold 4,095- and 31,998-token arrays
and 64 forced completion tokens for all four profiles:

| Profile | 4K PP / TPS | 32K PP / TPS | 32K peak GTT |
| --- | ---: | ---: | ---: |
| FP4 baseline | 187.44 / 11.75 | 122.61 / 10.70 | 14.02 GiB |
| FP4 MTP | 174.04 / 22.30 | 108.31 / 17.44 | 19.25 GiB |
| FP8 baseline | 188.90 / 7.60 | 119.58 / 6.87 | 26.00 GiB |
| FP8 MTP | 174.41 / 22.50 | 106.11 / 15.85 | 31.31 GiB |

FP8 baseline decode was about 35% slower than FP4. FP8 MTP was effectively tied
with FP4 MTP at 4K, lost 9.1% decode throughput at 32K, and used about 12 GiB
more GTT. FP8 therefore remains a quality-only experiment pending a fixed task
suite. FP4 MTP crossed FP4 baseline end-to-end time after about 42 generated
tokens at 4K and 954 at 32K on this repeated-context workload. These are
single-run screens; repeat finalists with the non-repeating pattern before
changing defaults. The machine summary is
[`results/qwen38-fp4-fp8-exact-context-2026-08-17.json`](results/qwen38-fp4-fp8-exact-context-2026-08-17.json).

The tuning helpers make these conclusions reproducible rather than relying on
hand comparison:

```bash
halo-ai tune mtp-compare \
  docs/results/qwen38-rocmfp4-baseline-2026-08-16.json \
  docs/results/qwen38-rocmfp4-mtp-2026-08-16.json \
  --output /var/opt/halo-ai/state/qwen38-mtp-comparison.json

halo-ai bench rocmfpx-context qwen38-27b-rocmfp4-mtp \
  --prompt-tokens 4095,31998 --completion-tokens 64 \
  --prompt-pattern unique --repetitions 3 --output RESULT.json
halo-ai tune context-compare BASELINE.json CANDIDATE.json
```

`mtp-compare` requires identical prompt-token sets and a matching target-model
SHA-256. It calculates same-host TTFT, prompt-speed, decode-speed, GTT, and
acceptance changes, then enforces the correctness and material-speed gates. It
does not consume performance claims from other hosts. The current records are
classified `hold-experimental`: decode is at least 10% faster at every measured
context, but strict cross-process token identity is not proven.

The pinned build advertises `ngram-mod,draft-mtp` and the suggested ngram
`24/48/64` settings, but the exact combination cannot start with
`--spec-mtp-strict-qwen`: ngram-mod disables recurrent rollback and strict-Qwen
MTP requires rollback covering the full draft. This was a clean model-load
rejection, not an OOM or device reset, and Halo does not expose the incompatible
combination as a profile.

External NPU and cross-version drafting remain deferred research. They are not
part of the active same-host FP4/FP8 GPU tuning loop.

### CachyOS Btrfs and Snapper policy

This host uses the standard CachyOS Btrfs separation. The following was
observed on 2026-08-09:

| Path | Backing filesystem/subvolume | Root Snapper coverage |
| --- | --- | --- |
| `/opt`, `/etc/opt`, `/var/opt`, `/var/lib` | root subvolume `@` | Yes, when the `root` Snapper config is active |
| `/srv` | separate subvolume `@srv` | No |
| `/var/cache` | separate subvolume `@cache` | No |
| `/home` | separate subvolume `@home` | No |
| `/boot` | separate VFAT filesystem | No |

The installed packages include `snapper`, `snap-pac`, `btrfs-assistant`, and
`limine-snapper-sync`. `snapper list-configs` confirmed configs named `root` for
`/`, `home` for `/home`, and `root-home` for `/root`. Reading their full creation
and retention policies still requires the operator's sudo password. Verify them
before enabling automation. Both `snapper-timeline.timer` and
`snapper-cleanup.timer` are currently enabled and active; the per-config
`TIMELINE_CREATE` setting determines which configs receive timeline snapshots.

```bash
sudo snapper -c root get-config
sudo snapper -c home get-config
sudo snapper -c root-home get-config
```

CachyOS states that `snap-pac` automatically creates a pre/post pair around
package installation, upgrade, and removal through `pacman`. Do not create an
extra manual pair around the same package transaction merely for halo-ai. Check
that the pair exists afterward:

```bash
sudo snapper list-configs
sudo snapper -c root list
```

Manual changes made by the halo-ai installer under `/opt`, `/etc/opt`,
`/var/opt`, or `/etc/default` are not pacman transactions. Wrap each coherent
host mutation in one descriptive root pre/post pair:

```bash
pre_id="$(sudo snapper -c root create \
  --type pre \
  --cleanup-algorithm number \
  --print-number \
  --description 'halo-ai: before install or host configuration')"

# Perform exactly one bounded install/configuration operation here.

sudo snapper -c root create \
  --type post \
  --pre-number "$pre_id" \
  --cleanup-algorithm number \
  --description 'halo-ai: after install or host configuration'
```

The real installer must create the matching post snapshot even when its
operation fails, label the outcome, and print both snapshot numbers. It should
refuse a host mutation if the root Snapper config is expected but unavailable,
unless the operator explicitly supplies `--no-snapshot`. Read-only commands,
container starts/stops, image pulls, cache creation, and model downloads do not
warrant root snapshots.

Root snapshots do not include `/srv`, which is desirable for multi-gigabyte
models: model replacement or deletion will not pin old model extents in routine
root snapshots. Treat models as separately managed data with recorded source,
size, and checksum; back up irreplaceable fine-tunes to another device. A Btrfs
snapshot on the same physical disk is not a backup. Do not add automatic
timeline snapshots for the model subvolume by default.

Boot-profile work requires two protections. A root Snapper pair covers
`/etc/default/limine` and other files under `/etc`, but it cannot cover this
host's VFAT `/boot`. Before regenerating Limine, the host-profile command must
also make and verify timestamped file backups of every `/boot` file it will
change. Never claim that a root snapshot alone can recover a damaged bootloader.

Before enabling automation, review retention in Btrfs Assistant or
`/etc/snapper/configs/root`. CachyOS currently recommends descriptive snapshots
and a small retained set (its guide recommends at most ten). The halo-ai script
will use the existing `number` cleanup policy and will not rewrite global
retention settings.

### Installing and resuming safely

`install.sh` deploys a content-addressed release beneath `/opt/halo-ai/releases`,
updates `current` and `/usr/local/bin/halo-ai` atomically, installs root-owned
catalog/preset templates, and grants the designated non-root operator ownership
only of mutable state/cache leaves. Preview it before the first run:

```bash
cd ~/Projects/source/ai/halo-ai
sudo ./install.sh --run-user "$USER" --dry-run
sudo ./install.sh --run-user "$USER"
```

The installer is a resumable state machine rather than one indivisible copy
operation. It records the release hash, operator, completed stages, failure
stage, and timestamp atomically in the root-only
`/var/lib/halo-ai-installer/install-state.env`. Its
stages are `directories`, `release`, `config`, `links`, and `verify`.

On a rerun for the same release/operator, each completed stage is verified
before being skipped and the first incomplete stage resumes. If a completed
stage has drifted, the installer fails closed and names the explicit
`--repair-stage STAGE` invocation; it does not silently overwrite an operator
fix. A journal belonging to different source content is archived only with
`--restart-journal`. Every attempt gets its own Snapper pre/post pair, including
failed attempts, and the failure handler prints the exact resume command. The
external model root is only validated for safe separation and is never copied,
downloaded into, chmodded, or chowned by the host installer.

One narrow cross-source exception supports installer bug fixes: if a failed
journal belongs to the same operator, its `release` stage is complete, and the
root-owned release's `.source-hash` still matches the journal, a corrected
installer may finish that pinned release without rewriting verified stages. It
prints the newer deferred release ID; after recovery, `--restart-journal` starts
a separate installation of that newer source content. If an attempt stopped in
`verify`, corrected installer source independently reruns the installed CLI check.
It pins and resumes the immutable release only when that check passes (for example,
when the failure was in installer stage dispatch). If the application check itself
fails, `--restart-journal` is required to archive the attempt and deploy corrected
source.
Uninstalling first is neither required nor recommended.

### Uninstalling the installed solution

The source checkout now includes `uninstall.sh`. It removes the system-integrated
halo-ai installation while treating the external model root as an immutable
exclusion. Always inspect its resolved plan first:

```bash
cd ~/Projects/source/ai/halo-ai
sudo ./uninstall.sh --run-user "$USER" --dry-run
sudo ./uninstall.sh --run-user "$USER"
```

The second command prints the same plan and requires the exact confirmation
`uninstall halo-ai`. Use `--yes` only in already-reviewed, non-interactive
automation. The script:

- Stops and removes rootless Podman containers and named volumes only when they
  carry `local.halo-ai.managed=true` in the selected operator's store.
- Removes `/usr/local/bin/halo-ai`, `/opt/halo-ai`, `/etc/opt/halo-ai`,
  `/var/opt/halo-ai`, `/var/cache/halo-ai`, and the root-only
  `/var/lib/halo-ai-installer` journal. The command link is removed only
  if it is a symlink resolving beneath `/opt/halo-ai`; all tree targets are
  fixed constants rather than configuration-controlled deletion paths. A target
  containing its own mountpoint is refused so a bind-mounted data tree cannot be
  traversed accidentally.
- Creates a `root` Snapper pre/post pair for covered system paths. Because
  `/var/cache` is a separate unsnapshotted subvolume on this host, its removal is
  explicitly shown in the confirmation plan. If Snapper is expected but
  unavailable, the script refuses the mutation unless `--no-snapshot` is
  supplied deliberately.
- Never removes, scans, changes ownership of, or changes permissions on
  `/srv/halo-ai/models`. It also protects a different absolute
  `HALO_AI_MODELS_ROOT` found in the installed configuration and refuses any
  removal target that overlaps either model root. A model artifact misplaced
  beneath an application/config/state/cache removal target also stops the
  uninstall so it can be relocated first.
- Leaves this source checkout, unrelated Podman objects, unlabeled legacy
  volumes, and shared/unlabeled images intact. `--remove-images` is opt-in and
  applies only to images that themselves carry the managed label.

Use `--keep-podman`, `--keep-config`, `--keep-state`, or `--keep-cache` for a
partial removal. `--keep-podman` and `--remove-images` are intentionally
mutually exclusive. If older manual testing created unlabeled
`halo-lemonade-*` volumes, the script reports and preserves them instead of
claiming ownership from their names alone.

## Phase 1: host preflight

Run these checks before pulling images or starting a service.

### Confirm the processor, GPU, and kernel driver

```bash
uname -r
lscpu | grep 'Model name'
lspci -nnk | grep -A4 -Ei 'vga|display|3d'
```

Expected results include Ryzen AI Max+ 395, Radeon 8060S/Strix Halo, and
`Kernel driver in use: amdgpu`.

### Confirm device access

```bash
ls -l /dev/kfd /dev/dri/renderD128 /dev/dri/card1
id
```

The current host makes `/dev/kfd` and `renderD128` accessible. If permissions
later change to `0660`, the user must have access through the host `render` and
`video` groups before a rootless container can use the devices.

### Confirm GPU memory, not just CPU-visible memory

```bash
for card in /sys/class/drm/card*/device; do
  if [[ -r "$card/mem_info_vram_total" ]]; then
    echo "$card"
    numfmt --to=iec --suffix=B < "$card/mem_info_vram_total"
    numfmt --to=iec --suffix=B < "$card/mem_info_gtt_total"
  fi
done

cat /proc/cmdline
cat /sys/module/ttm/parameters/pages_limit
cat /sys/module/ttm/parameters/page_pool_size
free -h
```

Expected on the current configuration:

```text
VRAM total: 2 GiB
GTT total:  118 GiB
```

The `halo-ai doctor` command obtains memory from the amdgpu sysfs
files. It must not reject this host based only on `/proc/meminfo` or `free`.

Treat unified-memory availability as three separate layers:

1. **Firmware topology:** the BIOS decides how much memory is fixed VRAM versus
   ordinary CPU-visible RAM.
2. **Kernel ceiling:** `amdgpu.gttsize` and TTM settings limit the dynamic GTT
   aperture.
3. **Application policy:** Lemonade or another runtime may filter models using
   only fixed VRAM unless it is configured to include GTT.

Record all three layers in `halo-ai doctor`. A plausible sysfs value is not a
substitute for confirming what Lemonade reports and successfully loading a model.
The current `ttm.page_pool_size` is `0` (driver default). Community guides differ
on whether to override it, while AMD's playbook abstracts the setting through
`amd-ttm`; do not change it automatically or merely to mirror another machine.

### Confirm software prerequisites

```bash
podman --version
curl --version
jq --version
vulkaninfo --summary
```

If anything is missing:

```bash
sudo pacman -S --needed podman curl jq pciutils vulkan-tools
```

Do not install host ROCm for this design. The selected images supply the ROCm
userspace matched to their applications; the host supplies the kernel and
`amdgpu` device interfaces.

### Confirm capacity and resource availability

```bash
df -h ~/.local/share ~/.cache
podman ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
ps -eo pid,comm,args | grep -E 'llama-|ds4-|lemond|vllm' | grep -v grep
```

Allow room for container images, Lemonade-managed downloads, the 269.54 GiB
external collection, and optional caches. The model tree is on `/srv`, but the
current rootless Podman store is under `/home`; check both filesystems. Stop
other GPU inference jobs before loading any large profile.

## Host-memory topology

AMD's playbooks recommend setting the BIOS dedicated GPU allocation to its
minimum and using a shared pool of at least 110 GiB for a 128 GiB Strix Halo.
That topology leaves close to the full physical memory CPU-visible and allows
the GPU to claim memory dynamically. The initial 96 GiB fixed-VRAM / 32 GiB
CPU-visible split failed this requirement: Qwen loading consumed about 24.1 GiB
of `gpu_active` pages, ordinary free memory fell to about 200 MiB, and the
kernel killed both llama-server and desktop Electron processes.

The corrected BIOS setting exposes about 123.5 GiB to Linux and 2 GiB as fixed
VRAM. The GTT/TTM ceiling was subsequently staged from 112 through 116 to the
currently verified 118 GiB. Keep this topology for Qwen, vision, and DeepSeek
testing unless new measurements justify a reviewed change.
The matching kernel values are exact:

| Aperture | `amdgpu.gttsize` (MiB) | `ttm.pages_limit` (4 KiB pages) | Antirez model-only headroom | Unsloth base headroom | Unsloth + DSpark headroom |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 112 GiB | 114688 | 29360128 | 21.11 GiB | 14.95 GiB | 4.80 GiB |
| 116 GiB | 118784 | 30408704 | 25.11 GiB | 18.95 GiB | 8.80 GiB |
| 118 GiB | 120832 | 30932992 | 27.11 GiB | 20.95 GiB | 10.80 GiB |

These are aperture-minus-file-size estimates, not promises of usable context:
runtime allocations, KV cache, compute buffers, the display, and other GPU work
consume the remainder. Raising the aperture increases the addressable ceiling;
it does not preallocate that amount or change the 128 GiB physical total. Do not
raise the aperture while firmware has carved most memory into fixed VRAM: that
reduces loader headroom instead of creating usable backing memory.

Changing the memory split or kernel command line is an administrator-controlled,
reboot-requiring operation. Normal `install`, `start`, and `stop` operations must
never change it. Only the explicit host-profile commands described below may
prepare a change, and they must not reboot automatically.

The completed 112 → 116 → 118 GiB progression used this reboot-delimited procedure:

1. Confirm the current AMD playbook still recommends the change.
2. Save the current `/proc/cmdline` and back up the Limine configuration.
3. Change the persistent `KERNEL_CMDLINE` entry in `/etc/default/limine`; do not
   edit generated `/boot/limine.conf` entries directly.
4. Run `sudo limine-mkinitcpio`.
5. Reboot.
6. Re-run the VRAM/GTT checks above.
7. Restore the previous configuration if GPU initialization or boot stability
   regresses.

The automation ceiling remains 118 GiB; 120 or 124 GiB requires a new explicit
operator decision and a higher configured ceiling.

See the current
[CachyOS boot-manager documentation](https://wiki.cachyos.org/configuration/boot_manager_configuration/)
before performing this procedure.

## Phase 2: Lemonade validation

Lemonade is the first deployment because a small model provides a fast end-to-end
test of container networking, persistent storage, GPU passthrough, backend setup,
and the OpenAI-compatible API.

### Image and persistent storage

Use AMD/Lemonade's official image:

```text
ghcr.io/lemonade-sdk/lemonade-server:latest
```

For a repeatable installation, `halo-ai install lemonade` resolves `latest` to
an immutable digest and records it in local state. `halo-ai update lemonade`
stages and records a newer digest without replacing an active container; perform
the profile smoke test before switching to it.

Pull the image and create volumes:

```bash
podman pull ghcr.io/lemonade-sdk/lemonade-server:latest
podman volume create \
  --label local.halo-ai.managed=true \
  --label local.halo-ai.component=lemonade-huggingface \
  halo-lemonade-huggingface
podman volume create \
  --label local.halo-ai.managed=true \
  --label local.halo-ai.component=lemonade-llama \
  halo-lemonade-llama
podman volume create \
  --label local.halo-ai.managed=true \
  --label local.halo-ai.component=lemonade-config \
  halo-lemonade-config
```

### Expose existing GGUFs without copying them

Lemonade officially supports a secondary `extra_models_dir`. It recursively
discovers `.gguf` files and presents them in the `custom` category with canonical
IDs prefixed by `extra.`.

Do not point it at `/srv/halo-ai/models`. That tree includes a ds4-specialized
model, AppleDouble files whose names end in `.gguf`, numbered shards, and
`mmproj`/`dspark` companions. A recursive filename scan cannot safely decide
which of those are runnable.

For a manual first test, expose only the two catalog-approved Qwen text mains:

```bash
export LEMONADE_QWEN35_FILE=/srv/halo-ai/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf
export LEMONADE_QWEN27_FILE=/srv/halo-ai/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q8_K_XL.gguf
export LEMONADE_QWEN35_BASENAME=${LEMONADE_QWEN35_FILE##*/}
export LEMONADE_QWEN27_BASENAME=${LEMONADE_QWEN27_FILE##*/}

for file in "$LEMONADE_QWEN35_FILE" "$LEMONADE_QWEN27_FILE"; do
  test -r "$file"
  test "$(head -c 4 "$file")" = GGUF
  stat --format='%s %n' "$file"
done
```

The lifecycle command generates an engine-specific allowlist view using
individual read-only bind mounts rather than copies. A long-lived Lemonade
container may receive both approved Qwen text mains so it can unload one and
load the other without recreation. It receives no projector, AppleDouble file,
ds4 model, or DeepSeek shard. Standalone engines still receive only the exact
files for one selected profile. The complete provider directory remains the
inventory source, not a runtime discovery input.

### Start Lemonade with ROCm

```bash
podman run -d \
  --name halo-lemonade \
  --label local.halo-ai.managed=true \
  --label local.halo-ai.component=lemonade \
  --device=/dev/kfd \
  --device=/dev/dri \
  -e LEMONADE_LLAMACPP=rocm \
  -p 127.0.0.1:13305:13305 \
  --mount "type=bind,src=$LEMONADE_QWEN35_FILE,dst=/models/extra/$LEMONADE_QWEN35_BASENAME,ro" \
  --mount "type=bind,src=$LEMONADE_QWEN27_FILE,dst=/models/extra/$LEMONADE_QWEN27_BASENAME,ro" \
  -v halo-lemonade-huggingface:/opt/lemonade/.cache/huggingface:U \
  -v halo-lemonade-llama:/opt/lemonade/llama:U \
  -v halo-lemonade-config:/opt/lemonade/.cache/lemonade:U \
  ghcr.io/lemonade-sdk/lemonade-server:latest
```

The server runs as an unprivileged user inside the official image. The `:U`
volume option gives that user ownership of the rootless Podman volumes. The host
port is deliberately restricted to `127.0.0.1` because the API should be treated
as unauthenticated unless an API key is explicitly configured.

### Observe startup and health

```bash
podman logs -f halo-lemonade
```

In a second terminal:

```bash
curl --fail http://127.0.0.1:13305/live
curl --fail http://127.0.0.1:13305/api/v1/models | jq
```

The web interface should be available at <http://127.0.0.1:13305>.

Keep the API namespaces distinct: application chat/model-list requests use the
AMD playbook's OpenAI-compatible base `http://127.0.0.1:13305/api/v1`, while
current Lemonade-specific management operations such as explicit load and unload
use `/v1/load` and `/v1/unload`.

### Enable and verify GTT-aware Lemonade configuration

Lemonade's documented default for `enable_dgpu_gtt` is `false`. On this unified
memory system it must be enabled so hardware-based model filtering includes the
GTT pool. Lock the llama.cpp backend to ROCm at the same time:

```bash
podman exec halo-lemonade \
  lemonade config set \
    enable_dgpu_gtt=true \
    llamacpp.backend=rocm \
    llamacpp.rocm_bin=latest \
    rocm_channel=stable \
    max_loaded_models=1 \
    no_broadcast=true \
    auto_check_model_updates=false \
    extra_models_dir=/models/extra

podman exec halo-lemonade lemonade config
podman exec halo-lemonade lemonade list | grep 'extra\.'
curl --fail http://127.0.0.1:13305/api/v1/system-info | jq
curl --fail http://127.0.0.1:13305/api/v1/models | jq \
  '.data[] | select(.id | startswith("extra."))'
```

Verify that the persisted configuration reports `enable_dgpu_gtt: true`, the
backend is `rocm`, `llamacpp.rocm_bin` is `latest`, the stable ROCm channel, one
loaded model maximum, discovery broadcasting disabled, and automatic update
checks disabled. System information
must enumerate the expected Radeon 8060S and large GPU-addressable pool. Treat a
small fixed-VRAM-only result as an application-configuration failure even when
sysfs reports 118 GiB of GTT.

The configuration resides in the persistent `halo-lemonade-config` volume, so it
survives container recreation. That volume is also Lemonade's executable/backend
cache. On the first validated gfx1151 ROCm install it held about 17 GiB: about
16 GiB beneath `bin/therock/gfx1151-7.13.0` and about 1.2 GiB beneath
`bin/llamacpp/rocm-stable`. Lemonade reuses each downloaded backend and TheRock
runtime on later starts; it downloads again only when the selected
backend/channel/version changes or the volume is removed. Normal `stop`, host
reinstall, and project upgrades preserve it. With the project default
`LEMONADE_LLAMACPP_ROCM_BIN=latest`, every explicit `halo-ai start` restarts the
small Lemonade supervisor before loading weights so Lemonade re-resolves the
newest stable-channel ROCm package. Existing package bytes remain cached. A
specific `bNNNN` value gives a reproducible pin; `builtin` follows the server
image's tested bundle.

Do not equate the package tag with the embedded binary's build string. Record
both. For example, a Lemonade package named `b10334` can legitimately contain a
binary reporting `b10333` when the packaging commit follows the llama.cpp
commit. `halo-ai start` records `package_version` from `lemonade backends` and
`binary_version` from the active `llama-server` path; `halo-ai test` also records
the API's `system_fingerprint`.

The three persistent Lemonade volumes have separate purposes:

| Volume | Container path | Purpose |
| --- | --- | --- |
| `halo-lemonade-config` | `/opt/lemonade/.cache/lemonade` | Configuration, pinned backend binaries, and TheRock ROCm runtime |
| `halo-lemonade-huggingface` | `/opt/lemonade/.cache/huggingface` | Models downloaded through Lemonade/Hugging Face |
| `halo-lemonade-llama` | `/opt/lemonade/llama` | Lemonade llama working data retained for compatibility |

The exact imported ID is an observed runtime
value: the launcher must compare the selected basename with the catalog and then
use the canonical `extra.*` ID returned by Lemonade. It must never select the
first recursively discovered file.

The launcher labels the long-lived container with a deterministic runtime-spec
fingerprint covering its image reference, devices, loopback port, persistent
volumes, exact allowlisted model/template mounts, and managed Lemonade settings.
It reuses the server for model switches only while that fingerprint matches;
catalog or mount changes recreate the container before loading a model.

Do not load the Qwen file yet. First complete the small canary below, then use
the explicit Qwen load policy that follows it. `halo-ai` must resolve the
canonical ID by matching Lemonade's returned checkpoint/path to the catalog
basename and require exactly one match; it must not guess an `extra.*` spelling
or select the first result.

An imported GGUF is not automatically guaranteed to support its architecture,
chat template, vision projector, or preferred backend. Treat discovery and a
successful load/inference request as separate acceptance checks.

For models pulled by Lemonade itself, the Hugging Face cache remains in
`halo-lemonade-huggingface`. If that cache later becomes large, it can be moved
to external storage with a dedicated writable bind mount or `HF_HOME`, but only
after confirming the filesystem supports Linux permissions, symlinks, atomic
renames, and adequate inference I/O. A read-only, profile-specific
`extra_models_dir` runtime view is the safer first deduplication mechanism for
the existing GGUFs.

### Load a small validation model

Use the model currently named by AMD's Lemonade playbook. Recheck the playbook if
the curated model name changes:

```bash
curl --fail-with-body \
  -X POST http://127.0.0.1:13305/v1/load \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"Gemma-4-E2B-it-GGUF"}'
```

Then make an OpenAI-compatible request:

```bash
curl --fail-with-body \
  http://127.0.0.1:13305/api/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Gemma-4-E2B-it-GGUF",
    "messages": [{"role": "user", "content": "Reply with: GPU test passed"}],
    "stream": false
  }' | jq
```

Confirm in the Lemonade logs or telemetry that the ROCm backend was selected. A
successful HTTP response alone is insufficient because a CPU fallback could also
produce a response.

### Load and switch the Qwen text baselines

Resolve and record the two canonical `extra.*` IDs by exact checkpoint basename.
Then load the 35B-A3B profile with fixed, inspectable settings. Lemonade exposes
context and backend as first-class load options; pass other supported
llama-server flags through `llamacpp_args`:

```bash
export LEMONADE_QWEN_MODEL_ID='extra.<exact-ID-returned-by-Lemonade>'
export LEMONADE_QWEN_ARGS='--flash-attn on --cache-type-k f16 --cache-type-v f16 --parallel 1 --batch-size 2048 --ubatch-size 512 --ctx-checkpoints 0 --cache-ram 0 --cache-reuse 0 --spec-type none'

jq -n \
  --arg model "$LEMONADE_QWEN_MODEL_ID" \
  --arg args "$LEMONADE_QWEN_ARGS" \
  '{
    model_name: $model,
    ctx_size: 32768,
    llamacpp_backend: "rocm",
    llamacpp_args: $args,
    merge_args: false,
    save_options: false,
    pinned: false
  }' |
  curl --fail-with-body \
    -X POST http://127.0.0.1:13305/v1/load \
    -H 'Content-Type: application/json' \
    --data-binary @-
```

At install/update time, render `llamacpp_args`, run the bundled backend's
`--help`, and reject unsupported or forbidden flags before loading. Lemonade
currently forbids supplying the model, port, context, GPU-layer count, Jinja
enablement, or projector through `llamacpp_args`; use its first-class fields and
catalog instead. Record the resolved Lemonade image, ROCm backend version, full
arguments, and `/api/v1/system-info` `recipe_options` in the trial.

The baseline deliberately uses:

- One slot, F16 K/V cache, Flash Attention, a 2,048 logical batch, and a
  conservative 512-token physical batch.
- Explicit 32,768 context rather than Lemonade's `-1` auto-resolution or the
  models' native 262,144-token maximum.
- Prompt caching left enabled, but cache reuse, context checkpoints, and the
  extra checkpoint RAM cache disabled until measurements justify them.
- `--spec-type none` explicitly. Lemonade recognizes the embedded MTP tensors
  and otherwise injects `draft-mtp` even for a profile intended as a baseline.
  Only the named MTP profiles may replace this with `draft-mtp`.
- The verified `load_mode=none` setting used by the implementation, which avoids
  the earlier mmap-related pressure path while retaining KV offload. Revisit
  mmap only as a separately measured profile.

This non-speculative configuration is a control, not the desired long-term
performance mode. After the corresponding named MTP profile passes capability,
quality, memory, restart, and repeated smoke tests, promote it to the preferred
Qwen profile and retain this baseline as the automatic/manual fallback. Apply
the same policy to the matching DeepSeek DSpark profile only after its unassisted
four-shard baseline passes. Never pair a DSpark companion with a different
checkpoint merely because it is available.

After the 512-token ubatch baseline is repeatable, benchmark 1,024 as one staged
change. Increase context through 65,536 and 131,072 in separate trials. Keep F16
KV for the quality/stability reference; a quantized-KV profile is experimental
and must pass output-quality and crash tests before use.

To switch Qwen models, call Lemonade's unload endpoint, poll system information
until the old backend PID disappears and VRAM/GTT returns near its pre-load
baseline, then load the other exact ID. Keep the Lemonade container running:

```bash
curl --fail-with-body \
  -X POST http://127.0.0.1:13305/v1/unload \
  -H 'Content-Type: application/json' \
  -d "{\"model_name\":\"$LEMONADE_QWEN_MODEL_ID\"}"

curl --fail http://127.0.0.1:13305/api/v1/system-info | jq \
  '.all_models_loaded // []'
```

This unload/load path is the normal Qwen profile switch. Recreate the container
only when its image, mounts, or server configuration changes. Stop the container
before starting ds4 or standalone llama.cpp.

When comparing performance, compare the same model and quantization on each
backend. Strix Halo decode is frequently memory-bandwidth-bound; Mixture-of-Experts
models can be much faster than similarly sized dense models because only a subset
of weights is active per token. Do not use a community throughput number from a
different model architecture as a health threshold.

### Stop and restart Lemonade

```bash
podman stop --time 60 halo-lemonade
podman start halo-lemonade
```

Downloaded models and backend artifacts remain in the named volumes. Removing
the container does not remove those volumes:

```bash
podman stop --time 60 halo-lemonade
podman rm halo-lemonade
```

Never remove the three `halo-lemonade-*` volumes as part of a normal stop or
update.

## Phase 3: ds4 deployment

ds4 is a deliberately narrow DeepSeek V4 engine, not a general GGUF runtime. It
accepts only layouts listed by its current upstream documentation. In
particular, the installed Unsloth IQ3_XXS shard set is not interchangeable with
the ds4 model even though both identify as `deepseek4`.

### Installed hybrid profile

The initial local profile uses the installed AMD-playbook hybrid model:

| Setting | Initial value |
| --- | --- |
| Profile | `ds4-deepseek-v4-flash-hybrid` |
| Image | `docker.io/kyuz0/strix-halo-ds4-toolbox:rocm-7.14` |
| Model | Antirez hybrid Q2/Q4 `fixed-0731`, 97.59 GB (90.89 GiB) |
| API model ID | `deepseek-v4-flash` |
| Context | 32,768 tokens |
| Prefill chunk | 2,048 tokens |
| Host endpoint | `http://127.0.0.1:8000/v1` |
| KV disk cache | Disabled |
| MTP/DSpark | Disabled |

The smaller approximately 80.8 GB IQ2_XXS layout recommended by the AMD
playbook remains a worthwhile future acquisition because it leaves more context
headroom. It is not currently installed and must not appear as the default.

The first live hybrid trial on this host completed in about 28 seconds at 32K.
ds4 reported ROCm on the Radeon 8060S, 90.88 GiB of resident model tensors, and
91.62 GiB planned for model, KV, and buffers. Sysfs observed about 107.6 GiB GTT
used and Linux retained about 9.7 GiB `MemAvailable`. A minimal request with
`reasoning_effort=none` returned the exact expected content. This is a successful
but tight baseline: do not raise its context or add speculative weights in the
same trial stage.

The installed hybrid remains deliberately at 32,768 context: the fixed matrix
left only about 8.9 GiB `MemAvailable`. Do not promote 65,536, 100,000, or
124,000-context hybrid profiles merely because the 32K smoke passed. Acquire and
verify the playbook-recommended smaller IQ2_XXS layout first, then test larger
contexts as separate boot-delimited profiles. Keep prefill chunk 2,048 initially
and stage 1,024 only after a classified prefill OOM.

### Validate and expose the installed model

```bash
export DS4_MODEL_FILE=/srv/halo-ai/models/antirez/deepseek-v4-gguf/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf

test -r "$DS4_MODEL_FILE"
test "$(stat --format='%s' "$DS4_MODEL_FILE")" -eq 97591747456
test "$(head -c 4 "$DS4_MODEL_FILE")" = GGUF
df -h "$DS4_MODEL_FILE"
```

The container receives this one file as `/models/model.gguf`. It does not receive
the adjacent AppleDouble sidecar or any Unsloth model. The lifecycle command
must never download over, rename, modify, or delete the externally supplied
file.

### Pull and start ds4

```bash
podman pull docker.io/kyuz0/strix-halo-ds4-toolbox:rocm-7.14
podman stop --time 60 halo-lemonade 2>/dev/null || true

podman run -d \
  --name halo-ds4 \
  --label local.halo-ai.managed=true \
  --label local.halo-ai.component=ds4 \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -p 127.0.0.1:8000:8000 \
  --mount "type=bind,src=$DS4_MODEL_FILE,dst=/models/model.gguf,ro" \
  docker.io/kyuz0/strix-halo-ds4-toolbox:rocm-7.14 \
  ds4-server \
    -m /models/model.gguf \
    --rocm \
    --host 0.0.0.0 \
    --port 8000 \
    --ctx 32768 \
    --prefill-chunk 2048
```

These device, IPC, capability, and seccomp options follow the container procedure
linked by the AMD ds4 playbook. Before first install or update, recheck its image
tag and arguments and record the resolved digest. Do not broaden the mount or
publish port 8000 on every interface.

### Observe and test ds4

```bash
podman logs -f halo-ds4
```

In another terminal, monitor allocation and then perform the smoke request:

```bash
watch -n 2 '
for f in /sys/class/drm/card*/device/mem_info_{vram,gtt}_used; do
  test -r "$f" && printf "%s: " "$f" && numfmt --to=iec --suffix=B < "$f"
done'

curl --fail http://127.0.0.1:8000/v1/models | jq
curl --fail-with-body \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Reply with: ds4 test passed"}],
    "reasoning_effort": "none",
    "stream": false
  }' | jq
```

Do not start Lemonade, llama.cpp, vLLM, or another large GPU job during this
trial. Record startup time, VRAM/GTT use after load, time to first token, and
generation tokens per second.

### Optional named KV disk-cache profile

The uncached baseline passed. The separate
`ds4-deepseek-v4-flash-hybrid-kv` profile enables an 8 GiB persistent disk KV
cache without changing the conservative baseline. The installer creates the
operator-private directory:

```bash
sudo install -d -o "$(id -un)" -g "$(id -gn)" -m 0700 \
  /var/cache/halo-ai/ds4-kv
```

The lifecycle renderer bind-mounts that directory at `/var/cache/ds4-kv` and
sets `--kv-disk-dir`, `--kv-disk-space-mb 8192`, minimum/cold/continued token
thresholds, 2,048-token alignment, and quantization mismatch rejection. The
cache accelerates restoration of repeated prompt prefixes and preserves session
checkpoints; it does not reduce active-context memory requirements. Use:

```bash
halo-ai start ds4-deepseek-v4-flash-hybrid-kv --switch
halo-ai test ds4-deepseek-v4-flash-hybrid-kv
halo-ai stop
```

The live cross-restart test used LongBench-v2 sample
`66f37eb9821e116aacb2d295`. Cold processing handled 10,278 prompt tokens at
97.37 PP tok/s, generated at 12.43 tok/s, took 107.7 seconds, and stored a
157.35 MiB entry for the aligned 10,240-token prefix. After recreating the
container, ds4 loaded that entry in 92.7 ms, processed only the 38-token suffix
at 25.01 tok/s, generated at 12.83 tok/s, and completed in 3.7 seconds with the
same response. The restored suffix PP rate must not be compared to full-prompt
PP; end-to-end latency and restored-token count are the useful cache measures.

Normal stop/restart preserves models and cache:

```bash
podman stop --time 120 halo-ds4
podman start halo-ds4
```

Configuration changes require container recreation, not model mutation. Cache
deletion requires a separate explicit maintenance command with confirmation.

## Phase 4: Seamless speech-to-speech translation

Speech is a separate, rootless ROCm service based on AMD's current Strix Halo
speech-to-speech playbook. It does not run inside Lemonade and does not share an
LLM container. The catalog pins `facebook/seamless-m4t-v2-large` at revision
`5f8cc790b19fc3f67a61c105133b20b34e3dcb76` and stores it directly at:

```text
/srv/halo-ai/models/facebook/seamless-m4t-v2-large
```

There is deliberately no intermediate `huggingface/` directory. The approved
download is about 9.25 GB and includes the two Transformers safetensors shards,
their index, and processor/tokenizer configuration. It excludes the repository's
duplicate pickle-based `.pt` checkpoints, reducing disk use and ensuring the
runtime loads only safetensors. Every selected file has an exact size and
published SHA-256 in the catalog and expected-file manifest.

The image follows the AMD gfx1151 Linux package track: PyTorch 2.12.0 with ROCm
7.14, torchvision 0.27.0, torchaudio 2.11.0, Transformers 4.57.1, and the
playbook's audio/processor dependencies. `halo-ai install speech` builds the
image locally and records its immutable image identity. Python packages and ROCm
wheels remain in Podman's image-layer cache, so unchanged rebuilds do not
redownload them. `halo-ai update speech` deliberately checks base/package
sources and builds a new image without mutating the active container.

Gradio is pinned to 6.16.0, the newest release whose published dependency range
overlaps Transformers 4.57.1's required Hugging Face Hub `<1.0`. Gradio 6.18+
requires Hub 1.x and cannot be resolved with AMD's pinned Transformers version;
the image build intentionally fails rather than forcing an incompatible pair.

Install the runtime and fetch only the pinned model subset:

```bash
halo-ai install speech
halo-ai models download seamless-m4t-v2-large
halo-ai models verify seamless-m4t-v2-large --full
```

The downloader resumes partial files in place and reuses already-complete files;
it never writes outside the exact external model directory. It requires the
catalog-pinned revision and allowlist instead of following a mutable repository
head. The ordinary uninstaller still preserves `/srv/halo-ai/models`.

Start the service, run the AMD sample through the API, and stop it with the same
lifecycle used for every other halo-ai runtime:

```bash
halo-ai start seamless-m4t-v2-large-speech --switch
halo-ai test seamless-m4t-v2-large-speech
halo-ai status
halo-ai stop
```

The UI is available only on this host at `http://127.0.0.1:7860/`. The health
endpoint is `http://127.0.0.1:7860/healthz`, and the multipart automation
endpoint is `POST /api/v1/translate` with an `audio` file and a target language
code such as `spa`. The smoke test downloads AMD's exact `input1.wav` once to
`/var/cache/halo-ai/speech`, verifies its size and SHA-256, requests Spanish
speech, and rejects an empty or malformed WAV. The model directory is mounted
read-only and only port 7860 on loopback is published.

The service does not maintain a hand-selected language shortlist. At startup it
intersects the pinned checkpoint's text-decoder, text-to-unit, and vocoder
language maps, then exposes all 36 languages that support speech input and
speech output through both `/healthz` and the Gradio dropdown. This includes
Ukrainian (`ukr`). A code absent from any required output stage is rejected
before inference rather than failing partway through synthesis.

Only one large GPU runtime is active at a time. `halo-ai start ... --switch`
stops an active LLM service before loading Seamless, and bare `halo-ai stop`
includes the speech container while retaining image layers, smoke input, and
external model weights.

Live validation on this host passed with PyTorch `2.12.0+rocm7.14.0`, HIP
`7.14.60850`, and the Radeon 8060S. The final cached image started and loaded
both local safetensors shards in about 8 seconds. AMD's pinned input produced a
valid 58,924-byte Spanish WAV in 7.04 seconds. Post-inference GTT use was 4.95
GiB and CPU `MemAvailable` was 109.5 GiB, confirming that the 118 GiB aperture
is compatible with this smaller service. Both the Gradio root page and health
endpoint returned HTTP 200 on loopback.

Transformers currently emits an upstream advisory that the constructed
Seamless decoder attention has no `layer_idx`; the tested forward/generation
path completed successfully. The project no longer uses the deprecated
`torch_dtype` loader argument—it uses `dtype`—and includes the small NUMA loader
dependency needed for a clean ROCm initialization.

## Phase 5: catalog-backed llama.cpp profiles

Lemonade remains the preferred common API for GGUF text and vision profiles.
Standalone llama.cpp is the controlled escape hatch for current upstream flags
that Lemonade does not expose or validate, including DeepSeek's DSpark
companion. It is also the fallback for embedded
MTP when Lemonade's bundled backend is too old or does not expose useful
speculative metrics.

The initial 2026-08-09 probe flattened only the two Qwen main files into the root
of `extra_models_dir` and did not mount `mmproj-F32.gguf`; Lemonade therefore
correctly discovered two text-only models. Lemonade 11.5.2 supports local
multimodal discovery when a main GGUF and its `mmproj` share a subdirectory. The
container now exposes separate read-only identities for the 27B text model and
the 27B vision pair. Both identities bind-mount the same main file, so no model
data is duplicated; only the `-vision` directory contains the projector.
Lemonade resolves that pair and supplies
`--mmproj` to its ROCm llama.cpp backend without copying either file into its
managed Hugging Face cache. Keep the standalone profile as a comparison and
fallback path.

### Findings from the existing Qwen launcher

`/home/michael/Projects/source/llama_llm/Serve-Llm.ps1` and its two Qwen 3.6
templates were reviewed read-only on 2026-08-09. They are useful design input,
not an installed dependency. The saved `llamacpp-help.txt` describes Windows
llama.cpp build `b9305`; it is a compatibility fixture, not evidence of the
flags in Lemonade's current container backend.

| Existing idea | halo-ai treatment |
| --- | --- |
| Named Qwen profiles and dry-run command display | Adopt; separate model, engine profile, and request preset, and add a shell-escaped `profiles render` command. |
| F16 K/V cache, Flash Attention, explicit batch/ubatch, one slot for MTP | Adopt as the stability baseline; benchmark ubatch and context one change at a time. |
| MTP guard against `-np > 1` and `--mmproj` | Adopt as a hard catalog validation rule. |
| Planner and implementer roles | Adopt as request presets on one loaded process when supported; never start two weight-resident servers by default. |
| Separate thinking/non-thinking templates | Catalog and hash both; mount them read-only, select the non-thinking-default variant explicitly, use `reasoning_effort=none` for off, and use request-level `chat_template_kwargs.enable_thinking=true` for on. |
| Recursive wildcard model selection | Reject; select exact catalog paths, sizes, digests, and companion roles. |
| `0.0.0.0` default and HIP-to-Vulkan fallback | Reject; bind loopback and fail closed on a backend mismatch. |
| Windows 64/64 split and `--no-mmap --no-kv-offload` HIP workaround | Do not transplant to CachyOS ROCm; this host has a different memory topology and scarce CPU-visible RAM. |

The PowerShell script also demonstrates why argument rendering needs tests. Its
`Get-BackendInfo` function reads that function's `$PSBoundParameters`, not the
script-level bound-parameter map, so explicit GPU-layer/Flash-Attention choices
can be overwritten and some optional arguments may not be emitted. Its 35B
profile is named MTP but defaults `SpecType` to `none`. halo-ai resolves all
overrides once into an immutable profile object, validate it, render the exact
command, and execute that same object—no second defaulting pass inside a helper.

The local templates are almost identical; the non-thinking version only adds a
default `enable_thinking=false`. halo-ai catalogs their exact sizes and SHA-256
digests as shared Qwen companion artifacts, mounts both read-only at
`/models/templates`, and explicitly gives the selected non-thinking-default file
to llama.cpp. A live test showed that the template default alone is not a
sufficient request-mode guarantee with current Qwen architecture handling.
Planner requests therefore explicitly send the still-supported request-level
`chat_template_kwargs.enable_thinking=true`, while implementer requests send
`reasoning_effort=none`. Test
system messages, native tools, nested JSON arguments, multi-step tool responses,
preserved reasoning, and empty content. Standalone llama.cpp's `/apply-template`
endpoint provides a no-inference fixture test; Lemonade requires a live tool-call
canary because its documented OpenAI layer does not promise every llama.cpp-only
request extension.

The selected standalone runtime is
`docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.14`. Upstream rebuilds this
stable tag from current llama.cpp master; `halo-ai install/update llamacpp`
pulls it and records the resolved immutable digest, while every trial records
the serving build fingerprint. The 2026-08-09 refresh resolved digest
`sha256:32d25e6f7608e1d221b71f51389c883afc655b9a3add9f7a787453dca288117b`
and llama.cpp build `b10335-74ce15741`. Re-run the capability canaries after a
digest change. The service binds to
`127.0.0.1:8080`, receives `/dev/kfd` and `/dev/dri`, and gets only the catalog
files required by its active profile as read-only mounts.

The initial execution-profile set is explicit:

| Profile ID | Engine | Model/feature set | Validation state |
| --- | --- | --- | --- |
| `qwen3.6-35b-a3b-q8xl-lemonade` | Lemonade | 35B-A3B text baseline | Passed |
| `qwen3.6-27b-q8xl-lemonade` | Lemonade | 27B text baseline | Passed |
| `qwen3.6-35b-a3b-q8xl-mtp-lemonade` | Lemonade | 35B-A3B MTP, text only | Passed; accepted draft tokens observed |
| `qwen3.6-27b-q8xl-mtp-lemonade` | Lemonade | 27B MTP, text only | Passed; accepted draft tokens observed |
| `qwen3.6-27b-q8xl-vision-lemonade` | Lemonade | 27B + installed F32 projector | Passed image canary |
| `qwen3.6-35b-a3b-q8xl-65k-lemonade` | Lemonade | 35B-A3B text, 65,536 context | Passed |
| `qwen3.6-35b-a3b-q8xl-128k-lemonade` | Lemonade | 35B-A3B text, 131,072 context | Passed |
| `qwen3.6-35b-a3b-q8xl-128k-mtp-lemonade` | Lemonade | 35B-A3B MTP, 131,072 context | Passed |
| `qwen3.6-27b-q8xl-128k-lemonade` | Lemonade | 27B text, 131,072 context | Passed |
| `qwen3.6-27b-q8xl-128k-vision-lemonade` | Lemonade | 27B vision, 131,072 context | Passed image canary |
| `qwen3.6-27b-q8xl-vision-llamacpp` | llama.cpp | 27B + installed F32 projector | Passed image canary |
| `qwen3.6-27b-q8xl-mtp-llamacpp` | llama.cpp | 27B embedded MTP, text only | Passed; accepted draft tokens observed |
| `qwen3.6-35b-a3b-q8xl-mtp-llamacpp` | llama.cpp | 35B-A3B embedded MTP, text only | Passed; accepted draft tokens observed |
| `deepseek-v4-flash-0731-iq3xxs-llamacpp` | llama.cpp | Four-shard base model | Passed; high-memory |
| `deepseek-v4-flash-0731-iq3xxs-dspark-llamacpp` | llama.cpp | Four shards + exact DSpark companion | Passed; accepted draft tokens observed; tight |

`ds4-deepseek-v4-flash-hybrid` is defined in Phase 3. The two Lemonade MTP
profiles append `--spec-type draft-mtp --spec-draft-n-max 2` and otherwise reuse
their verified baseline load options. They are enabled only when Lemonade's
bundled llama.cpp advertises both flags and metrics prove nonzero drafted and
accepted tokens. Standalone MTP is also a validated comparison path. DSpark
uses explicit `--spec-type draft-dspark` plus the exact cataloged companion; its
canary must likewise report nonzero drafted and accepted tokens.
Catalog compatibility is a tested assertion, not an assumption based on the
`.gguf` suffix.

Sampling and reasoning are request presets, not model-server identities. Begin
with Qwen's current upstream sampling recommendations and make the non-thinking
mode explicit so client defaults cannot change it:

| Request preset | Thinking | Temperature | Top-p | Top-k | Min-p | Presence penalty | Repeat penalty |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3.6-planner-coding` | On | 0.6 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| `qwen3.6-implementer` | Off | 0.7 | 0.8 | 20 | 0.0 | 1.5 | 1.0 |

Current llama.cpp server documentation still supports the per-request
`chat_template_kwargs` object, including `{"enable_thinking":false}`. What was
deprecated is using `enable_thinking` through the server-startup
`--chat-template-kwargs` route. The current first-class per-request off switch is
`"reasoning_effort":"none"`; other `reasoning_effort` values presently do not
alter reasoning. Consequently the implementer preset and the default non-thinking
smoke test send `reasoning_effort: none`, while the planner sends request-level
`chat_template_kwargs.enable_thinking: true`. The canary calls the active
llama.cpp `/apply-template` endpoint inside the container: off must render the
closed empty `<think>…</think>` prefix and on must not. This is deterministic;
requiring non-empty `reasoning_content` from a trivial completion is not, because
an enabled model can answer without using a reasoning trace. Keep the two pinned
Jinja files because they still define formatting and conversation semantics. Test
`{"preserve_thinking":true}` separately for long agent sessions. Do not silently
pretend the role changed. Because upstream warns that high presence penalties
can cause language mixing or reduced quality, retain a separate
`presence_penalty=0` implementer benchmark before making 1.5 universal.

### LongBench-v2 at 128K

Use LongBench-v2 after the profile smoke matrix, not as the first stability
test. The project pins `zai-org/LongBench-v2` revision
`2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9` and its `data.json` artifact:

```text
15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2 465490535 data.json
```

The 503-sample dataset spans short, medium, and long contexts, with some inputs
far beyond 128K. halo-ai uses the upstream zero-shot prompt and its strict
`The correct answer is (A-D)` extractor. It asks the active llama.cpp backend's
`/v1/chat/completions/input_tokens` endpoint to count the fully rendered chat
request, so the pinned Qwen Jinja template and special tokens are included.
The usable input budget is the profile context minus `--max-tokens`.

Dataset bytes are stored once below
`/var/cache/halo-ai/benchmarks/longbench-v2/<revision>/`. Results and their run
manifest are under `/var/opt/halo-ai/state/benchmarks/longbench-v2/` unless
`--output` is supplied. Each completed or skipped sample is appended and
`fsync`ed before the next begins. Repeating an identical command resumes by
sample ID; changing the profile, filters, overflow policy, output budget, or
selected IDs against an existing output fails closed.

Run a six-cell canary (one deterministic sample for each available
difficulty/length combination), then the full native-fit subset:

```bash
halo-ai start qwen3.6-35b-a3b-q8xl-128k-lemonade
halo-ai bench longbench-v2 download
halo-ai bench longbench-v2 run qwen3.6-35b-a3b-q8xl-128k-lemonade
halo-ai bench longbench-v2 run qwen3.6-35b-a3b-q8xl-128k-lemonade --suite full
```

The default `--overflow fit` records over-budget samples as
`skipped_overflow`; its accuracy is therefore a native-128K compatible-subset
score, and the reported coverage is part of the result. It must not be called
the official overall LongBench-v2 score. The upstream reference runner instead
keeps the first and last halves of over-limit token sequences. To measure that
separate policy, halo-ai middle-truncates only the document while preserving the
question and choices, repeatedly measures the real rendered token count, and
marks every affected record:

```bash
halo-ai bench longbench-v2 run qwen3.6-35b-a3b-q8xl-128k-lemonade \
  --suite full --overflow middle
```

This is a clearly labeled truncation-compatible score, not evidence that the
model consumed the original 2M-word context. Use `--length`, `--difficulty`,
`--domain`, `--limit`, and `--sample-id` for diagnostic slices. A minimal
cross-profile comparison should use the same exact sample ID and a separate
result file per profile, for example:

```bash
halo-ai bench longbench-v2 run PROFILE \
  --sample-id 66f37eb9821e116aacb2d295
```

LongBench remains a text request when run against a vision profile; it measures
that profile's loaded text path and memory overhead but does not exercise the
projector. Pair it with `halo-ai test VISION_PROFILE`, whose generated-image
canary is the projector acceptance check.

ds4 does not expose a token-count endpoint. halo-ai therefore permits only an
explicit `--sample-id` DS4 run, applies a conservative preflight estimate, and
requires the completion's `usage.prompt_tokens` as the authoritative recorded
count. DS4 canary/full and middle-truncation modes fail closed. Because ds4 does
not return a `timings` object either, halo-ai normalizes the latest completed
request's `prompt done` and generation-rate server log lines into PP/TPS fields
in the benchmark record; absence of matching lines remains `null` rather than a
fabricated estimate.

Use `halo-ai bench
longbench-v2 score RESULTS.jsonl` to recompute coverage and accuracy without a
running model. Preserve the manifest, JSONL, score JSON, active image digest,
and backend build together when comparing plain, MTP, or DSpark runs.

Use this staged validation order, completing the API smoke test and rebooting
before advancing after any failure or material setting change:

1. Run `qwen3.6-35b-a3b-q8xl` text-only at 32,768 context. Its MoE layout is the
   best first local performance/coding candidate; the repository's vision
   projector is not installed.
2. Run `qwen3.6-27b-q8xl` text-only at 32,768 context.
3. Add the installed `mmproj-F32.gguf` to a separate Qwen 27B vision profile and
   validate a small image request.
4. Enable embedded Qwen MTP in a separate text-only Lemonade profile when its
   bundled backend passes the flag check; otherwise use standalone llama.cpp with
   `--spec-type draft-mtp --spec-draft-n-max 2`. Current upstream guidance says
   MTP cannot be combined with `--mmproj` or parallel slots (`-np > 1`), so the
   profile validator must reject those combinations.
5. Run `deepseek-v4-flash-0731-iq3xxs` with all four ordered shards at 32,768
   context and no speculative companion. The 104.21 GB model already exceeds
   fixed VRAM and consumes most of the shared aperture.
6. Only after that baseline is repeatable, try the matching
   `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf` companion. Main plus companion is
   about 107.20 GiB, leaving only about 4.8 GiB of the observed aperture before
   runtime and context allocations; treat this as a high-lockup-risk experiment.

The approved 116 and 118 GiB stages raise that model-only remainder to about 8.8
and 10.8 GiB, respectively, but do not make DSpark a safe baseline. On
2026-08-09 the operator explicitly selected 118 GiB as the common retest ceiling
for Qwen long-context, vision, speech, and DeepSeek workloads. This supersedes
the earlier automatic-tuning preference to try 116 first; the mutation remains
guarded, snapshot-backed, and reboot-delimited.

The Unsloth DSpark file is checkpoint-specific and belongs only to the Unsloth
0731 shard set. It is not the smaller support file currently documented by ds4
and must never be paired with the Antirez hybrid model or passed to ds4.

The cold fixed-sample result explains why DSpark remains optional rather than
the DeepSeek default. Baseline PP/TPS were 56.78/10.21 tok/s at 97.3 GiB GTT;
DSpark measured 36.43/15.31 tok/s at 107.5 GiB. Its accepted draft tokens improve
decode, but the extra model and slower prefill made the 10,254-input-token,
33-output-token request take 283.7 seconds instead of 183.9 seconds. Prefer the
plain profile for long-prompt/short-answer work and benchmark DSpark for
output-heavy generation before selecting it.

### Deferred vLLM profiles

vLLM is not a baseline engine for the installed collection. Its current GGUF
path is highly experimental, underoptimized, and provided through the separate
`vllm-gguf-plugin`; a tokenizer from the original model is also recommended.
The AMD integrated-GPU vLLM playbook does not currently validate these exact
DeepSeek V4 Flash or Qwen 3.6 checkpoints.

If vLLM is added later, prefer original Hugging Face Transformers/safetensors
weights under a distinct `/srv/halo-ai/models/huggingface/` tree or shared HF
cache and verify the exact architecture against vLLM's current support table.
Do not duplicate hundreds of gigabytes merely to check a box: acquire native
weights only for a benchmark or feature vLLM is expected to improve. Use a
digest-pinned rootless container, a read-only weights mount, writable cache
outside the source tree, and loopback port 8001. Keep
`VLLM_GGUF_EXPERIMENTAL=0` unless an explicit experimental profile is selected.

## Deferred NPU profile

The reviewed community setup demonstrates that FastFlowLM on the XDNA NPU can
serve a small model alongside GPU inference with limited GPU throughput impact.
That is potentially valuable for background workloads, but it is deliberately
out of scope for the initial GPU deployment.

There is a configuration conflict to resolve first:

- The current GPU profile uses `amd_iommu=off` for performance.
- The reviewed NPU setup requires IOMMU and a working `amdxdna` device.
- This host currently has no `/dev/accel/accel0` and no bound NPU driver.
- The guide's `apt`, PPA, and `amdxdna-dkms` commands target Ubuntu and must not
  be run on CachyOS.

If NPU support is later requested, create a separate, explicit host profile:

1. Recheck AMD's current Ryzen AI Linux and Lemonade NPU playbooks.
2. Snapshot the working GPU baseline, image digests, kernel, firmware, and
   throughput measurements.
3. Determine the supported CachyOS/Arch driver and firmware pairing; never copy
   NPU firmware manually without its matching driver.
4. Re-enable the required IOMMU mode through Limine and reboot.
5. Require `amdxdna` to bind cleanly and `/dev/accel/accel0` to exist before
   attempting an NPU container.
6. Validate NPU execution from response metadata, expected throughput, and zero
   GPU activity rather than trusting a utilization gauge alone.
7. Re-run the GPU canary and compare against the saved baseline.

The future lifecycle script may eventually add a `npu` profile, but it must not
silently alter IOMMU, kernel modules, firmware, memlock limits, or boot settings.

### Reboot-required IOMMU host profiles

The lifecycle tooling implements an explicit IOMMU toggle because disabling
the AMD IOMMU has produced measurable GPU-inference and memory-bandwidth gains on
Strix Halo community benchmarks. This is a workload tradeoff, not a universal
optimization:

| Profile | Persistent kernel setting | Result |
| --- | --- | --- |
| `gpu` | Add `amd_iommu=off`; remove `iommu=pt` | Best known GPU-only performance; XDNA NPU unavailable |
| `npu` | Remove `amd_iommu=off` and `iommu=pt` | Use the kernel's translated IOMMU default; GPU remains available and XDNA can initialize |

Do not set `amd_iommu=on`: it is not a documented AMD IOMMU kernel value. NPU
mode normally means removing the explicit `off` override while ensuring IOMMU is
enabled in firmware. A future validated `iommu=pt` option may be added separately,
but it should not be conflated with the conservative NPU profile.

The implemented interface is:

```text
halo-ai host-profile status
halo-ai host-profile init
halo-ai host-profile set gpu
halo-ai host-profile set gpu --gtt-gib 118
halo-ai host-profile set npu
halo-ai host-profile rollback [backup-id]
```

`set` accepts `--gtt-gib 112|116|118`; it changes the matching
`amdgpu.gttsize` and `ttm.pages_limit` pair atomically. `init`, `set`, and
`rollback` also accept `--dry-run`, `--yes`, and the explicit
escape hatch `--no-snapshot`. Always run the dry run as the ordinary operator;
run an accepted mutation through `sudo halo-ai ...` only afterward.

Behavior and safeguards:

1. `status` is read-only. It reports the running `/proc/cmdline`, persistent
   Limine setting, whether a reboot is pending, IOMMU initialization evidence,
   `amdxdna` binding, and `/dev/accel/accel0` availability.
2. `init` is required once if `/etc/default/limine` does not exist, as on the
   current host. It derives a proposed `KERNEL_CMDLINE[default]` from the complete
   running command line, displays it, and requires explicit confirmation before
   writing. It must preserve root UUID, Btrfs subvolume, GTT/TTM, and unrelated
   arguments byte-for-byte.
3. `set` changes only the exact `amd_iommu=off` and `iommu=pt` tokens. It refuses
   malformed or duplicate IOMMU arguments and never reconstructs the rest of the
   command line from defaults.
4. Before every write, create a timestamped, checksummed copy of
   `/etc/default/limine` plus every current regular `/boot` file beneath
   `/var/opt/halo-ai/state/host-profiles/backups/<backup-id>/`, and show the
   old/new command-line diff. Capacity is checked before the boot copy begins.
5. Apply the accepted change with `sudo limine-mkinitcpio`. Never edit generated
   `/boot/limine.conf` entries directly.
6. Do not reboot automatically. Print the expected post-reboot checks and mark
   the profile as pending until the running kernel matches the persistent setting.
7. `rollback` restores a selected verified backup, runs
   `sudo limine-mkinitcpio`, and likewise requires a manual reboot.
8. Before selecting `npu`, warn that the BIOS IOMMU setting must be enabled. The
   script cannot change firmware settings.
9. Before selecting `gpu`, warn that `/dev/accel` will disappear and refuse the
   change while an NPU workload is active.

Post-reboot verification for `gpu`:

```bash
grep -w 'amd_iommu=off' /proc/cmdline
test ! -e /dev/accel/accel0
```

Post-reboot verification for `npu`:

```bash
! grep -qw 'amd_iommu=off' /proc/cmdline
find /sys/kernel/iommu_groups -mindepth 1 -maxdepth 1 -type d -print -quit |
  grep -q .
sudo dmesg | grep -i -E 'AMD-Vi|IOMMU'
lspci -nnk -d 1022:17f0
test -e /dev/accel/accel0
```

The NPU profile is not healthy merely because `amd_iommu=off` is absent. It is
healthy only after IOMMU initializes, `amdxdna` binds, the accelerator node
exists, and the matched driver/firmware/runtime validation succeeds.

## Lifecycle command

`bin/halo-ai` is a Bash entry point backed by a Python standard-library
implementation. It currently provides the following interface:

```text
halo-ai doctor
halo-ai models <scan|list|show|verify> [model-id] [--full]
halo-ai profiles <list|show|render> [profile-id]
halo-ai presets <list|show|render> [preset-id]
halo-ai install [lemonade|llamacpp|ds4|vllm|all]
halo-ai start <profile-id> [--switch]
halo-ai stop [profile-id|all]
halo-ai restart <profile-id>
halo-ai status
halo-ai env
halo-ai logs [profile-id] [-f]
halo-ai test [profile-id] [--preset preset-id]
halo-ai update <lemonade|llamacpp|ds4|vllm|all>
halo-ai tune <status|discard>
halo-ai host-profile <status|init|set|rollback> [gpu|npu|backup-id]
```

### Configuration

The system-installed script reads `/etc/opt/halo-ai/config.env`, followed by
an optional operator override at
`${XDG_CONFIG_HOME:-$HOME/.config}/halo-ai/config.env`. An explicit
`--config PATH` overrides both. The example configuration will define:

```bash
HALO_AI_RUN_USER=michael
HALO_AI_STATE_DIR=/var/opt/halo-ai/state
HALO_AI_CACHE_DIR=/var/cache/halo-ai
HALO_AI_MODELS_ROOT=/srv/halo-ai/models
HALO_AI_CATALOG_DIR=/etc/opt/halo-ai/models.d
HALO_AI_PRESET_DIR=/etc/opt/halo-ai/request-presets.d
HALO_AI_INVENTORY_FILE=/var/opt/halo-ai/state/model-inventory.json
HALO_AI_MODELS_REQUIRE_MOUNT=0
HALO_AI_MODELS_EXPECT_UUID=
HALO_AI_DEFAULT_PROFILE=qwen3.6-35b-a3b-q8xl-lemonade

LEMONADE_IMAGE=ghcr.io/lemonade-sdk/lemonade-server:latest
LEMONADE_PORT=13305
LEMONADE_VALIDATION_MODEL=Qwen3-0.6B-GGUF
LEMONADE_LLAMACPP_ROCM_BIN=latest

# Stable gfx1151 tag; each pull and trial records its resolved digest/build.
LLAMACPP_IMAGE=docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.14
LLAMACPP_PORT=8080

DS4_IMAGE=docker.io/kyuz0/strix-halo-ds4-toolbox:rocm-7.14
DS4_PORT=8000
DS4_KV_CACHE_ENABLED=0
DS4_KV_CACHE_DIR=/var/cache/halo-ai/ds4-kv
DS4_KV_CACHE_MB=8192

# Deferred and disabled until an explicit profile is installed.
VLLM_IMAGE=
VLLM_PORT=8001
VLLM_GGUF_EXPERIMENTAL=0

# GTT observation and iterative tuning policy.
HALO_AI_GTT_TARGET_GIB=auto
HALO_AI_GTT_AUTOTUNE=stage
HALO_AI_GTT_CANDIDATES_GIB=112,116,118
HALO_AI_GTT_MAX_GIB=118
HALO_AI_OS_RESERVE_GIB=4
```

Set `HALO_AI_MODELS_REQUIRE_MOUNT=1` when the model root is backed by a separate
or removable filesystem. If `HALO_AI_MODELS_EXPECT_UUID` is also set, `start`
must compare it with `findmnt -no UUID -T "$HALO_AI_MODELS_ROOT"` and fail
closed on a mismatch.

The script will parse configuration as data rather than source arbitrary shell
content. It will reject relative model paths, nonnumeric ports/context sizes,
unreadable model files, unsafe paths such as `/` or the whole home directory,
and GTT targets outside the physically safe range. Model paths, context,
companions, expected sizes, and engine compatibility belong in the catalog, not
in free-form environment variables. A profile may override an engine default,
but environment input may not turn an unapproved companion or engine pairing
into a valid profile.

The GTT candidate list must be strictly increasing, include the observed
baseline, contain only whole GiB values no larger than the configured maximum,
and preserve the OS reserve. On this host the accepted path is
`112 -> 116 -> 118`; it is not synthesized from an arithmetic step.

### GTT environment and OOM feedback loop

`HALO_AI_GTT_TARGET_GIB` is the operator-facing desired aperture. It accepts
`auto` or a whole number of GiB. `auto` means use the observed aperture initially
and let recorded evidence inform later recommendations; it does not mean silently
rewrite boot configuration after an error.

Before every `doctor`, `status`, `start`, and `test`, the script will regenerate
`${XDG_RUNTIME_DIR}/halo-ai/hardware.env` atomically from sysfs. `halo-ai env`
will print the same shell-safe values so an interactive shell can import them
with `source <(halo-ai env)`:

```bash
HALO_AI_GTT_APERTURE_BYTES=126701535232
HALO_AI_GTT_APERTURE_GIB=118
HALO_AI_GTT_USED_BYTES=112635904
HALO_AI_VRAM_TOTAL_BYTES=2147483648
HALO_AI_VRAM_USED_BYTES=916910080
HALO_AI_GTT_TARGET_GIB=auto
HALO_AI_GTT_PENDING_GIB=
```

`HALO_AI_GTT_APERTURE_GIB` is observed, read-only state. Setting it manually must
not influence decisions; the script always replaces it from
`mem_info_gtt_total`. `HALO_AI_GTT_TARGET_GIB` is configuration and may influence
recommendations. Keep the byte values for exact calculations and expose GiB only
for operator readability.

The tuning-policy values are:

| `HALO_AI_GTT_AUTOTUNE` | Behavior after a classified memory failure |
| --- | --- |
| `observe` | Record the failure and measurements only |
| `suggest` | Record it and print the proposed next-boot change |
| `stage` | Persist one bounded adjustment for a new boot; default |

Before model loading begins, create and `fsync` a durable trial record containing
the current boot ID from `/proc/sys/kernel/random/boot_id`, the full workload
fingerprint, memory snapshot, and intended settings. Mark the trial successful
only after the model loads and the API smoke test completes. This pre-write is
essential because a GPU-memory lockup may prevent the process from catching the
OOM or flushing logs.

Every caught OOM, allocation failure, or unfinished prior-boot trial will append
one JSON object to `$HALO_AI_STATE_DIR/oom-history.jsonl` containing:

- Timestamp, profile, engine, image digest, model identity and size.
- Context, prefill chunk, backend, and optional speculative-decoding settings.
- Exact VRAM/GTT totals and usage immediately before start, at peak if sampled,
  and after failure.
- CPU `MemAvailable`, exit status, relevant container log excerpt, and OOM source
  (`kernel`, `ROCm`, `ds4`, `Lemonade`, `llama.cpp`, `vLLM`, or `unknown`).
- The selected adjustment, whether it requires reboot, and the result of the
  next attempt using that adjustment.
- Trial boot ID, completion state, and whether the evidence is a caught OOM or a
  suspected lockup inferred from an unfinished record after the boot ID changed.

The adjustment order is intentionally conservative:

1. Correct an application-policy error first, including missing Lemonade
   `enable_dgpu_gtt=true`, an unexpected CPU fallback, or mmap duplication when
   a GGUF exceeds CPU-visible RAM but fits GPU-addressable memory. For current
   llama.cpp builds, stage `--load-mode none` rather than the deprecated
   `--no-mmap`, and apply it only to a future-boot trial.
2. For a caught prefill OOM, stage a smaller engine-appropriate prefill/batch
   setting; for ds4 move to `2048`, then `1024` on a later failed boot-delimited
   iteration.
3. Stage a 20% context reduction, rounded down to a multiple of 4096, but never
   below 32768 without an explicit operator override.
4. Treat an unfinished prior-boot trial as a suspected lockup. Stage only a
   pressure-reducing change: disable optional MTP/DSpark, reduce prefill chunk,
   or reduce context. Never respond to a lockup by automatically increasing GTT.
5. Consider a larger GTT aperture only for a cleanly caught, classified memory
   failure when allocation approached at least 90% of an applicable VRAM or GTT
   pool and earlier pressure-reducing iterations did not resolve the workload.
   Do not recommend a GTT increase for either Qwen baseline unless telemetry
   actually shows GTT exhaustion; their weights fit the fixed VRAM pool.
6. In `auto`, stage the next strictly larger value in
   `HALO_AI_GTT_CANDIDATES_GIB`: 116 GiB, then 118 GiB on this host. Cap it by
   `HALO_AI_GTT_MAX_GIB` and physical unified memory minus
   `HALO_AI_OS_RESERVE_GIB`. An explicit numeric target is a ceiling, never a
   guarantee that the script will allocate that amount. Values not present in
   the approved candidate list require an explicit configuration change.

A staged adjustment is always a future-boot iteration, including changes that
could technically be made without rebooting. Store staged runtime settings in
`$HALO_AI_STATE_DIR/pending-trial.env` rather than overwriting the operator's
configuration. Refuse to execute that trial while its originating boot ID is
still current. After a manual reboot, display the pending diff and use it for one
explicit `start`.

For a GTT adjustment, the script must not resize root-writable TTM sysfs
parameters live. It stages matching `amdgpu.gttsize` MiB and
`ttm.pages_limit` 4 KiB-page values through the guarded Limine host-profile
mechanism, reports `HALO_AI_GTT_PENDING_GIB`, and waits for a manual reboot.
After reboot it verifies the observed aperture before allowing the pending trial.

The feedback loop must terminate. Permit zero same-boot automatic retries and one
staged change for the same workload fingerprint until its result is reviewed.
Never convert an unclassified ordinary crash, GPU reset, driver fault, planned
shutdown, or corrupted-model error into a memory-tuning action. An unfinished
trial without corroborating OOM evidence remains labeled `suspected_lockup`, not
`confirmed_oom`, even though its next stage is intentionally safer.

### Command behavior

- `doctor` performs read-only host, firmware-topology, kernel GTT/TTM,
  application-memory, permissions, port, storage, NPU-state, and container
  checks. It prints CPU-visible RAM separately from VRAM and GTT, warns about
  executable/world-readable model files, and verifies that a rootless container
  can read the selected files.
- `models scan` discovers candidate files without changing them. `models verify`
  checks catalog structure and sizes, while `--full` computes and persists local
  SHA-256 results. `models list/show` reports catalog identity, companions,
  verification state, compatible engines, and the reason a profile is disabled.
- `profiles list/show` reports the exact model, engine, context, arguments,
  mounts, endpoint, and risk tier without starting a container. `profiles
  render` prints the exact shell-escaped container command or Lemonade load JSON
  that `start` would execute, with secrets redacted.
- `presets` lists and renders request-level reasoning and sampling fields. It
  validates that the active engine supports every requested extension and never
  silently drops reasoning, template, tool, or vision fields.
- `install` pulls images, records immutable digests, creates only project-owned
  volumes/directories, and performs no host-driver or boot-loader changes. It is
  distinct from the root-run `install.sh` that deploys a static release
  under `/opt`; only that host installer uses the manual Snapper policy.
- The root-run `uninstall.sh` removes the static installation and only labeled
  project containers/volumes. It protects the configured external model root,
  uses a Snapper pair for covered paths, preserves images by default, and offers
  explicit keep flags for partial removal.
- `start` resolves a catalog profile, validates every required shard and
  companion, and refuses to run when another large inference profile is active.
  It writes the durable trial record before load, constructs minimal individual
  read-only model mounts, and refuses a same-boot retry after memory failure.
- When switching between Qwen Lemonade profiles with a pinned backend, `start`
  reuses the healthy Lemonade container and unloads the old model first. Under
  the default `latest` policy it restarts the lightweight supervisor on every
  explicit start so Lemonade checks its channel before loading weights. A
  cross-engine switch still stops the old container.
- `start --switch` gracefully stops the active profile, waits for its process to
  exit and release GPU allocations, and then starts the selected profile.
- `stop` defaults to `all` and stops every labeled halo-ai runtime container;
  passing a profile ID limits it to that profile's engine. It also cancels an
  in-progress model load without classifying the operator cancellation as an
  inference failure. The command is idempotent and preserves stopped
  containers, images, volumes, external models, caches, and downloaded runtime
  components. Removing those persistent objects belongs to `uninstall.sh`, not
  the normal runtime lifecycle.
- `status` reports container state, endpoint health, active backend, configured
  image tag/digest, current VRAM/GTT use, target aperture, and pending aperture.
- `env` refreshes and prints the sourceable, read-only hardware snapshot described
  above.
- `tune status` reports the active trial, last completed trial, pending adjustment,
  originating boot ID, and whether a reboot boundary has been crossed.
- `tune discard` removes a pending runtime adjustment only after archiving the
  reason; it does not undo a persistent Limine change.
- `logs` delegates to `podman logs` and supports follow mode.
- `test` performs the relevant model-list and minimal chat request.
- `update` pulls the current configured tag, records old and new digests and the
  rollback image reference, and leaves active containers unchanged. The operator
  then renders/recreates the selected profile and runs its smoke test explicitly;
  a 90+ GiB model load is never an implicit side effect of an image pull.
- `host-profile` is the only command family authorized to prepare persistent
  IOMMU changes. It follows the backup, diff, Limine regeneration, manual-reboot,
  and post-reboot verification rules above.

All commands will use fixed container and volume names prefixed with `halo-` and
will operate only on resources carrying a `local.halo-ai.managed=true` label.

### Failure and rollback rules

- Never silently fall back from GPU to CPU. Treat backend mismatch as a failed
  health check.
- Never infer validity from a `.gguf` suffix. Ignore AppleDouble `._*` files,
  reject non-GGUF magic, and refuse incomplete or misordered shard sets.
- Never launch `mmproj` or `dspark` as a main model. Reject a companion whose
  catalog checkpoint does not exactly match the selected main model.
- Never combine Qwen MTP with a vision projector or more than one parallel slot
  while current upstream limitations remain in force.
- Never report MTP active merely because its flags were requested. Require the
  backend version/help check plus speculative metrics showing drafted and
  accepted tokens.
- Never apply a planner/implementer preset if the engine ignores its thinking or
  template fields. Fail the capability canary and route to a supported profile.
- Require Lemonade to persist `enable_dgpu_gtt=true` and confirm the expected
  pool through `/api/v1/system-info` before using large-model eligibility as an
  acceptance signal.
- On OOM, stop the failed container, preserve logs, and recommend lowering ds4
  context before suggesting host-memory changes. Stage the bounded adjustment
  for a future boot and do not retry during the current boot.
- If ds4 prefill OOMs, try `--prefill-chunk 2048`; do not assume this changes the
  context limit.
- If a new image fails, recreate the container from the previously recorded
  digest without changing persistent data.
- If a port is occupied, identify the owning process and stop. Do not kill an
  unrelated process automatically.
- If device permissions fail, report the exact node and host ownership. Do not
  work around the issue with a privileged container.
- Never remove external models, Lemonade volumes, or the ds4 KV cache during
  ordinary rollback. Labeled volumes and cache are removed only by the explicit
  uninstaller unless its corresponding keep option is set; external models are
  excluded even then.
- Never delete AppleDouble sidecars or chmod/chown model files as a side effect
  of scan, verify, doctor, start, rollback, or update.
- Never enable the vLLM GGUF plugin as an implicit fallback; it requires an
  explicitly named experimental profile.
- Never report an IOMMU profile switch as complete before reboot. A mismatch
  between persistent Limine configuration and `/proc/cmdline` is `pending`, not
  `active`.
- Never report a requested GTT size as active until
  `mem_info_gtt_total` confirms it after reboot. Kernel-command-line values,
  observed aperture, target aperture, and pending aperture are distinct states.
- Never increase GTT automatically in response to an unfinished trial or system
  lockup; reduce workload pressure first and require clean caught-OOM evidence for
  a later aperture increase.
- Never stage 120 or 124 GiB while the configured and operator-approved ceiling
  remains 118 GiB.

## Acceptance sequence

The installed solution remains acceptable when it can complete this sequence:

1. Host preflight reports `gfx1151`, working `amdgpu`, approximately 123.5 GiB
   CPU-visible RAM, 2 GiB fixed VRAM, 118 GiB GTT, and the current NPU/IOMMU
   state without treating the NPU as a GPU blocker. It rejects an aperture that
   lacks CPU-visible backing.
2. `models scan` reports nine real GGUF files totaling 289,414,525,472 bytes,
   ignores all six AppleDouble sidecars as models, groups the four DeepSeek
   shards in order, and classifies `mmproj`/`dspark` as companions.
3. Fast verification matches all nine exact published byte sizes. A fixture for
   `models verify --full` distinguishes published expected hashes from locally
   computed and verified hashes without loading the model into memory.
4. A missing shard, wrong-sized file, non-GGUF `._*` file, mismatched companion,
   or unreadable rootless-container mount disables the affected profile with a
   specific error and does not mutate the model tree.
5. Lemonade starts on loopback, persists `enable_dgpu_gtt=true`, reports the
   expected memory pool through `/api/v1/system-info`, and selects ROCm.
6. The small curated Lemonade model returns a valid chat completion.
7. Lemonade's allowlist view contains exactly the two Qwen text mains—no
   projector, sidecar, or DeepSeek file—and resolves both canonical `extra.*`
   IDs by checkpoint basename rather than list order.
8. The Qwen 35B-A3B and 27B text baselines each load at 32,768 context with one
   slot, F16 KV, Flash Attention, the declared batch/ubatch, ROCm, and
   `--spec-type none`. The trial record exactly matches Lemonade's reported
   backend PID and recipe options.
9. Switching between Qwen baselines unloads the prior backend and returns GPU
   allocation near baseline before loading the next, without recreating the
   Lemonade container or changing either model file.
10. Planner and implementer request fixtures prove thinking on/off, sampling,
    native tool calls, nested arguments, and multi-step tool responses. An
    unsupported reasoning/template toggle fails the deterministic rendered-
    template check rather than being silently ignored.
11. A Qwen MTP profile rejects `--mmproj` and `-np > 1`; when enabled, backend
    metrics show nonzero drafted and accepted tokens. The Qwen 27B projector
    works only in its separate non-MTP vision profile.
12. The Antirez hybrid passes exact path, size, GGUF, and catalog validation; ds4
   loads it at 32,768 context with prefill chunk 2,048 and no other inference
   runtime active.
13. ds4 `/v1/models` and `/v1/chat/completions` return valid responses, then ds4
    stops without modifying the model or adjacent sidecar.
14. The four-shard Unsloth DeepSeek baseline is recognized as 104,207,848,032
    bytes and 1,328 tensors and launches without DSpark. Its separate DSpark
    profile uses `draft-dspark`, mounts only the exact 10,896,057,440-byte
    companion, and must report nonzero drafted and accepted tokens; it can never
    target ds4 or the Antirez model.
15. Each runtime stops cleanly, retains its intended cache, and permits a
    different profile to start only after GPU allocations are released. No
    service listens on a non-loopback host address.
16. `host-profile status` correctly identifies the current GPU profile and can
    distinguish a staged profile from the running profile without changing it.
17. `halo-ai env` reports the observed 118 GiB aperture from sysfs, ignores a
    forged `HALO_AI_GTT_APERTURE_GIB`, and keeps target/pending values distinct.
18. A simulated Qwen OOM does not propose a larger aperture without measured GTT
    exhaustion. A qualifying DeepSeek OOM stages 116 GiB
    (`118784`/`30408704`), and only a reviewed later-boot failure may stage 118
    GiB (`120832`/`30932992`); 120/124 remain out of bounds.
19. A simulated model OOM produces one history record and one bounded suggestion;
    it neither retries in the same boot nor changes Limine in `observe` or
    `suggest` mode.
20. A fixture containing an unfinished trial from a previous boot is classified
    as `suspected_lockup`, stages a lower-pressure configuration, and never stages
    a GTT increase.
21. The host installer deploys an immutable release under `/opt/halo-ai`, keeps
    model and cache bytes out of the source tree, and can switch back to the
    preceding `current` target without a root snapshot rollback.
22. A simulated host install creates a labeled Snapper pre/post pair; package-only
    installation relies on the verified `snap-pac` pair instead of duplicating
    it, and Limine tests prove that `/boot` backups are handled separately.
23. An uninstaller fixture removes only exact installed paths and managed-label
    Podman objects, preserves unlabeled/shared objects and image layers by
    default, and proves that sentinels under both the default and a configured
    external model root remain byte-for-byte unchanged. Unsafe command links,
    overlapping roots, and misplaced model artifacts fail closed.
24. LongBench-v2 download verifies the pinned revision, byte count, and SHA-256;
    the active llama.cpp backend counts the rendered Jinja request, native-fit
    mode records overflows without truncation, and an interrupted or completed
    canary resumes without duplicating sample IDs.
25. The DS4 disk-KV profile writes only beneath its 8 GiB private cache, stores
    a qualifying repeated prefix, restores it after container recreation, and
    leaves the uncached baseline profile unchanged.
26. The Seamless downloader selects exactly the 12 files at the pinned revision
    beneath `/srv/halo-ai/models/facebook`, full verification passes, ROCm health
    names the expected GPU/HIP versions, the UI stays on loopback, and AMD's
    sample produces a valid translated WAV.

The cataloged baselines, MTP, vision, DSpark, ds4 disk cache, Seamless speech,
and 128K Qwen profiles have now passed. Test any new context, KV-cache policy,
image digest, model revision, or future vLLM profile one change at a time with
the staged reboot policy. The
measured production preference is MTP for Qwen. For the installed DeepSeek
IQ3_XXS checkpoint, use the plain profile for prefill-heavy work and reserve
DSpark for output-heavy workloads whose own benchmark justifies its 10+ GiB
memory cost; both remain available as comparison and recovery paths.

## References

- AMD: [Getting Started with Lemonade](https://developer.amd.com/playbooks/lemonade-getting-started/)
- AMD: [Running DeepSeek V4 Flash with ds4](https://developer.amd.com/playbooks/deepseek-v4-flash-ds4/)
- AMD: [Real-Time Speech-to-Speech Translation](https://developer.amd.com/playbooks/speech2speech-translation/)
- AMD: [vLLM inference](https://developer.amd.com/playbooks/vllm-inference/)
- Lemonade: [Running Lemonade in Docker](https://lemonade-server.ai/docs/guide/install/docker/)
- Lemonade: [Arch installation](https://lemonade-server.ai/docs/guide/install/arch/)
- Lemonade: [Server configuration](https://lemonade-server.ai/docs/guide/configuration/)
- Lemonade: [Custom models](https://lemonade-server.ai/docs/guide/configuration/custom-models/)
- Lemonade: [Embeddable model organization](https://lemonade-server.ai/docs/embeddable/models/)
- Lemonade: [Server API](https://lemonade-server.ai/docs/api/lemonade/)
- Lemonade: [OpenAI-compatible API](https://lemonade-server.ai/docs/api/openai/)
- llama.cpp: [Server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- LongBench-v2: [dataset](https://huggingface.co/datasets/zai-org/LongBench-v2)
- LongBench-v2: [official runner](https://github.com/THUDM/LongBench)
- ds4 upstream: [antirez/ds4](https://github.com/antirez/ds4)
- ds4 Strix Halo toolbox: [kyuz0/strix-halo-ds4-toolbox](https://github.com/kyuz0/strix-halo-ds4-toolbox)
- Antirez: [DeepSeek V4 GGUF](https://huggingface.co/antirez/deepseek-v4-gguf)
- Meta: [Seamless M4T v2 Large](https://huggingface.co/facebook/seamless-m4t-v2-large)
- Unsloth: [DeepSeek V4 Flash 0731 GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)
- Unsloth: [Qwen3.6 27B MTP GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF)
- Unsloth: [Qwen3.6 35B-A3B MTP GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)
- vLLM: [GGUF quantization](https://docs.vllm.ai/en/latest/features/quantization/gguf/)
- vLLM: [Supported models](https://docs.vllm.ai/en/latest/models/supported_models/)
- CachyOS: [Filesystem and default Btrfs subvolumes](https://wiki.cachyos.org/installation/filesystem/)
- CachyOS: [Btrfs snapshots](https://wiki.cachyos.org/configuration/btrfs_snapshots/)
- CachyOS: [Boot Manager Configuration](https://wiki.cachyos.org/configuration/boot_manager_configuration/)
- freedesktop.org: [Filesystem Hierarchy Standard](https://specifications.freedesktop.org/fhs/latest-single/)
- Snapper: [command reference](https://snapper.io/manpages/snapper.html)
- Podman: [rootless setup and storage configuration](https://github.com/containers/podman/blob/main/docs/tutorials/rootless_tutorial.md)
- Linux kernel: [Kernel command-line parameters](https://docs.kernel.org/admin-guide/kernel-parameters.html)
- AMD XDNA: [Linux driver](https://github.com/amd/xdna-driver)
- Community field report: [burrellka/MyStrixHaloSetup](https://github.com/burrellka/MyStrixHaloSetup)
- Strix Halo GPU tuning: [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)
