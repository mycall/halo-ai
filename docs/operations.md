# Daily operations and runtime maintenance

[Documentation index](README.md) · [Project README](../README.md)

For first-time setup, see [Installation](installation.md) and [Profiles](profiles.md).

## Start, inspect, switch, and stop

```bash
halo-ai doctor
halo-ai profiles list
halo-ai profiles show qwen3.8-fn
halo-ai profiles render qwen3.8-fn
halo-ai start qwen3.8-fn --switch
halo-ai test qwen3.8-fn
```

`profiles render` previews the container command without starting it. `start`
loads a real runtime and requires its artifacts to be acquired first. `--switch`
stops a conflicting managed runtime before starting the requested one. Only one
large managed inference runtime runs at a time.

Manage runtime resources with:

```bash
halo-ai status
halo-ai stop                 # stop every halo-ai runtime
halo-ai stop PROFILE         # stop the engine used by one profile
halo-ai restart PROFILE
```

`halo-ai stop` is safe to repeat and also cancels an in-progress model load. It
stops only containers labeled as managed by this project. External models,
container images, named cache/config volumes, and downloaded runtime components
are preserved so the next start does not reinstall them. Use `uninstall.sh`
when the installed solution itself should be removed.

## Connect a client

Use the endpoint for the selected runtime; they do not all share one port.
These are the checked-in defaults and can be overridden in configuration.

| Runtime | Local endpoint | Guide |
| --- | --- | --- |
| Halogen | `http://127.0.0.1:8731/v1` | [API and model IDs](halogen-flash.md#run-and-select-a-profile) |
| DS4, ROCmFPX, Strix Vulkan | `http://127.0.0.1:8000/v1` | [DS4 API](ds4.md), [Qwen and OpenCode](qwen-profiles.md#27b-aliases-vision-dflash2-and-opencode) |
| Lemonade | `http://127.0.0.1:13305/api/v1` | [Lemonade setup](#lemonade-and-standalone-llamacpp) |
| Standalone llama.cpp | `http://127.0.0.1:8080/v1` | [Qwen runtime profiles](qwen-profiles.md) |
| Speech | `http://127.0.0.1:7860` | [UI and translation API](speech.md) |

## Redeploy checkout changes

After editing or updating the source checkout, redeploy it with the Fish- and
Bash-friendly reload helper. It determines the operator account, invokes sudo,
archives the prior installer journal, and skips the confirmation prompt:

```bash
./reload.sh
# Optional non-mutating preview:
./reload.sh --dry-run
```

## Inspection and maintenance commands

| Command | Effect |
| --- | --- |
| `halo-ai logs PROFILE -f` | Follow the selected engine's container logs |
| `halo-ai env` | Refresh and print the observed hardware snapshot |
| `halo-ai presets list` / `show PRESET` / `render PRESET` | Inspect request sampling and reasoning policy |
| `halo-ai install ENGINE` | Prepare the configured runtime image and project storage |
| `halo-ai update ENGINE` | Prepare the configured runtime image and record rollback information; leave active containers unchanged |
| `halo-ai tune status` | Inspect trial state, pending adjustment, and reboot boundary |
| `halo-ai tune discard` | Archive/discard a pending runtime adjustment; does not undo persistent boot settings |

Runtime `install` differs from root-run `install.sh`, which deploys the host
application. An image update does not implicitly load a model; recreate the
selected service explicitly and smoke-test it. Use `halo-ai --help` and
subcommand `--help` for complete arguments.

## Lemonade and standalone llama.cpp

Acquire a catalog profile before starting it. A text baseline example is:

```bash
halo-ai profiles acquire qwen3.6-35b-a3b-q8xl-lemonade
halo-ai start qwen3.6-35b-a3b-q8xl-lemonade --switch
halo-ai test qwen3.6-35b-a3b-q8xl-lemonade
curl --fail http://127.0.0.1:13305/api/v1/models
curl --fail http://127.0.0.1:13305/api/v1/system-info
```

Lemonade needs `enable_dgpu_gtt=true`, backend `rocm`, and one loaded model.
Inspect its persisted configuration with `podman exec halo-lemonade lemonade config`.
A fixed-VRAM-only pool or CPU backend fails the large-model acceptance check.
The manager registers approved files in place and isolates text and vision
companions; no duplicate copy of external GGUFs is needed.

Standalone llama.cpp uses separate catalog profiles and port 8080. Select an
explicit `-llamacpp` profile from the catalog, then use the same acquire/start/test
sequence. Its models route is `http://127.0.0.1:8080/v1/models`. See
[Qwen compatibility](qwen-profiles.md) before combining MTP, vision, or drafts.

## Runtime pins, caches, and compatibility

Halo supports one Lemonade server release: 11.8.1, pinned by its immutable
amd64 image-manifest digest. The former 11.7 pin and the floating `latest`
reference are deprecated migration inputs, not selectable defaults. The
first Lemonade ROCm start downloads a llama.cpp backend and TheRock runtime.
Backend/runtime data remains in the historical `halo-lemonade-config` cache
volume, Hugging Face downloads use `halo-lemonade-huggingface`, and 11.8's
persistent JSON state uses the `halo-lemonade-state` volume. Before 11.8
migrates JSON from `.cache` to `.config`, `halo-ai install/update lemonade`
makes a content-addressed backup under
`/var/opt/halo-ai/state/lemonade-config-backups`.

Rootless Podman retains pulled image layers in its content-addressed local
store, and Halo does not prune them during install or update. Advancing to a
later 11.8.x pin therefore reuses every common image layer. The four named
volumes are independent of the image layer store, so backend/runtime downloads,
registered state, and Hugging Face models also survive container replacement.

The default retains the qualified stable backend package
`LEMONADE_LLAMACPP_ROCM_BIN=b10597`; the server release and its downloaded
llama.cpp backend are independent pins. Existing downloads and the large
TheRock runtime are reused. Change the package or channel only for an explicit
compatibility trial, then restore the qualified pin.

Standalone llama.cpp defaults to a locally built, provenance-labeled image
containing Unsloth build 10715 with qwen4exp external-head MTP support. Its
runtime base remains Kyuz0's immutable ROCm 10.0 Strix Halo image; both the
base digest and Unsloth release archive SHA-256 are checked before use.

Lemonade 11.8.1 also exposes an experimental native DS4 backend using the same
`b0001` gfx1151 artifact Halo already pins. Halo keeps its direct DS4 engine as
the qualified default. A trial with the installed mixed-precision hybrid showed
that Lemonade forces SSD streaming and its ROCm prefill fails before the first
token (`selected expert id -1` at layer 3), including with the maximum accepted
expert cache. The native profile is recorded but disabled; direct DS4 remains
the working path for DSpark, 384K context, disk-KV, provenance, and timings.
This is the exact mixed-quant ROCm streaming defect tracked in
[DS4 #896](https://github.com/antirez/ds4/issues/896); Lemonade's unconditional
streaming policy and missing full-residency opt-out are tracked in
[Lemonade #3431](https://github.com/lemonade-sdk/lemonade/issues/3431).

## Memory failures and recovery

Before loading, the manager writes a durable trial record with the boot ID,
workload identity, and memory/settings snapshot. Classified memory failures
are recorded in `oom-history.jsonl`. An unfinished trial from a prior boot is a
suspected lockup, not automatically a confirmed OOM.

Inspect `halo-ai logs PROFILE`, `halo-ai status`, and `halo-ai tune status` after
a failure. There are no automatic same-boot retries. Pending settings live in
`pending-trial.env` and are applied to an explicit trial after a manual reboot.
An ordinary crash, GPU reset, corrupted artifact, or operator stop does not
justify a memory-tuning change.

`HALO_AI_GTT_AUTOTUNE` accepts `observe` (record only), `suggest` (also propose),
and `stage` (persist a bounded future-boot adjustment, the default). The policy
reduces workload pressure first: optional speculation, prefill/batch size, or
context. Suspected lockups never trigger an automatic GTT increase. A larger
aperture requires classified exhaustion evidence, the configured candidate
limits, and the guarded [host-profile workflow](host-configuration.md#gpu-and-npu-boot-profiles).
`auto` does not authorize silent boot edits or live sysfs resizing.

For a bad runtime image, use the recorded previous digest and explicitly
recreate/test the profile. Preserve models, named volumes, and caches during
ordinary rollback. For boot settings, use `host-profile rollback BACKUP_ID`
and reboot; discarding a runtime trial does not roll back Limine.
