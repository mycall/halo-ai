# halo-ai

`halo-ai` is a guarded lifecycle manager for AMD Strix Halo inference services
on CachyOS. It keeps external models in `/srv/halo-ai/models`, runs GPU services
through rootless Podman, and separates static installation, configuration,
state, and cache data according to the filesystem hierarchy.

The detailed design and operations runbook is in
[`docs/halo-ai.md`](docs/halo-ai.md).

## Validate the checkout

```bash
./tests/container.sh
./bin/halo-ai doctor
./bin/halo-ai models verify
./bin/halo-ai profiles list
```

`tests/container.sh` runs the source/unit/shell suite in a network-disabled,
read-only rootless Podman container with the checkout mounted read-only and
only an ephemeral `/tmp` writable. It installs no host Python packages. The
host CLI checks below remain read-only except when an explicit lifecycle verb
such as `install`, `start`, or `stop` is requested.

These commands do not load a model. `models verify --full` additionally hashes
about 278 GiB of cataloged model files and writes an inventory record, so it is
deliberately not part of the smoke test.

## Preview and install

```bash
sudo ./install.sh --run-user "$USER" --dry-run
sudo ./install.sh --run-user "$USER"
```

After editing or updating the source checkout, redeploy it with the Fish- and
Bash-friendly reload helper. It determines the operator account, invokes sudo,
archives the prior installer journal, and skips the confirmation prompt:

```bash
./reload.sh
# Optional non-mutating preview:
./reload.sh --dry-run
```

The host installer is resumable. It journals each verified stage in the
root-only `/var/lib/halo-ai-installer/install-state.env`. After a failure, fix the reported problem
and rerun the same command; completed stages are verified and skipped. Use
`--repair-stage NAME` only when the installer reports drift in a completed
stage.

After host installation, run lifecycle commands as the configured rootless
operator:

```bash
halo-ai doctor
halo-ai install lemonade
halo-ai profiles render qwen3.6-35b-a3b-q8xl-lemonade
```

To stage the approved 118 GiB shared-memory ceiling with an exact, backed-up
Limine edit:

```bash
halo-ai host-profile set gpu --gtt-gib 118 --dry-run
sudo halo-ai host-profile set gpu --gtt-gib 118 --yes
```

Reboot manually and require `halo-ai host-profile status` to report both
`running_gtt_gib` and `persistent_gtt_gib` as `118` before long-context tests.

Rendering is safe and read-only. `halo-ai start PROFILE` starts a real GPU
runtime and may consume most unified memory; follow the staged procedure in the
runbook first.

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
[`docs/results/qwen3.8-rocmfp4-baseline-2026-08-16.json`](docs/results/qwen3.8-rocmfp4-baseline-2026-08-16.json).
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
[`docs/results/qwen3.8-rocmfp4-mtp-2026-08-16.json`](docs/results/qwen3.8-rocmfp4-mtp-2026-08-16.json).

The already-present, hash-verified ROCmFP8 artifact is exposed only through the
explicit `qwen3.8-27b-rocmfp8-baseline` and `qwen3.8-27b-rocmfp8-mtp` GPU
profiles. A matched single-run 4K/32K screen found no FP8 performance reason to
change defaults: FP8 baseline decode was about 35% slower than FP4, and FP8 MTP
was tied at 4K but 9.1% slower at 32K than FP4 MTP while using about 12 GiB more
GTT. Keep FP8 for a future fixed quality comparison. The aligned record is
[`docs/results/qwen3.8-fp4-fp8-exact-context-2026-08-17.json`](docs/results/qwen3.8-fp4-fp8-exact-context-2026-08-17.json).

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

For a long run under controlled office power, use the profile matrix. It runs
the source suite in its read-only, network-disabled Podman test container, then
starts, tests, benchmarks, and stops each ready profile sequentially. The first
run uses the checked-in same-host wall-time seeds; every later run sorts fastest
to slowest using successful elapsed times in `profile-history.tsv`:

```bash
# After installing the current checkout, run every ready profile.
tests/office-profile-matrix.sh --scope all

# Restrict a run to the active Qwen3.8 and DS4 optimization profiles.
tests/office-profile-matrix.sh --scope optimize
```

ROCmFPX profiles receive the exact-token, non-repeating 4K/32K benchmark with
three repetitions. Other LLM engines receive the fixed LongBench-v2 systems
sample; speech receives its modality smoke test. Results, environment snapshots,
per-profile logs, and ordering history are stored under
`/var/opt/halo-ai/state/benchmarks/office-profile-matrix`. Use `--smoke-only`
for a quick orchestration rehearsal. A cleanup trap stops managed inference
containers after normal completion, failure, or interruption.

The runner holds a transient systemd sleep inhibitor for the complete matrix
and independently detects any suspend interval from the boottime/monotonic
clock offset. Use `--exclude-profiles CSV` to preserve an explicit audit trail
when a suspected profile should not be retried unattended.

The runner selects `performance` with `powerprofilesctl` and requires both the
power-profiles daemon and ACPI platform profile to remain in that state before
and after every profile. Because the installed kernel does not expose a numeric
GPD WIN 5 DPTC limit, the runner does not equate the profile name with a watt
cap. Instead it samples the kernel's AMDGPU PPT sensor once per second and, for
a non-smoke run, requires the observed run peak to exceed 45 W. Override only
the observation threshold with `--minimum-ppt-watts N`; raw samples and the run
peak are retained with the results.

Halo supports one Lemonade server release: 11.8.1, pinned by its immutable
amd64 image-manifest digest. The former 11.7 pin and the floating `latest`
reference are deprecated migration inputs, not selectable defaults. The
first Lemonade ROCm start downloads a llama.cpp backend and TheRock runtime.
Backend/runtime data remains in the historical `halo-lemonade-config` cache
volume, Hugging Face downloads use `halo-lemonade-huggingface`, and 11.8's
persistent JSON state uses the new `halo-lemonade-state` volume. Before 11.8
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

Standalone llama.cpp now defaults to Kyuz0's supported `rocm-10.0` Strix Halo
image, pinned at the qualified registry manifest digest. Explicit update trials
may follow the rebuilt upstream tag, but normal installs and starts reuse the
audited ROCm 10.0 / llama.cpp build 10711 artifact.

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

## DS4 disk KV cache

The original DS4 hybrid profile remains the uncached control. Use the separate
cache profile when coding agents or other clients repeatedly send a long shared
prefix:

```bash
halo-ai install ds4
halo-ai start ds4-deepseek-v4-flash-hybrid-kv --switch
halo-ai test ds4-deepseek-v4-flash-hybrid-kv
```

For the qualified 384K DSpark/Think Max configuration, `ds4` is a catalog alias
for the full profile name:

```bash
halo-ai start ds4 --switch
halo-ai test ds4 --preset deepseek-v4-think-max
```

The Qwen3.8 lanes have explicit short aliases:

```bash
halo-ai start qwen3.8df2 --switch  # Q6 XL vision + DFlash2, native 262K
halo-ai start qwen3.8-27b-q6xl-vision-lemonade --switch  # Same Q6/BF16 target on ROCm/HIP
halo-ai start qwen3.8fp4 --switch
halo-ai start qwen3.8fp8 --switch
```

`qwen3.8df2` uses the separately pinned Strix Vulkan llama.cpp fork, not
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

The checked-in OpenCode configuration exposes this alias as
`halo-ai/qwen3.8df2`, alongside `ds4`, `qwen3.8fp4`, and `qwen3.8fp8` under one
provider. These mutually exclusive managed LLM runtimes share the loopback
endpoint on port 8000, with text and image inputs enabled for DFlash2.
All three Qwen3.8 entries default to `medium` reasoning and expose OpenCode
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
[`docs/results/qwen3.8-q6xl-vulkan-rocm-ab-2026-08-23.json`](docs/results/qwen3.8-q6xl-vulkan-rocm-ab-2026-08-23.json)
for the exact runtime identities and samples.
The deprecated Lemonade 11.7 qualification record repeated the vision canary in three fresh processes
and reran both target-only controls without prompt caching. ROCm median
prefill/decode was 130.07/8.013 tok/s versus Vulkan 99.69/8.466 tok/s. See
[`docs/results/qwen3.8-q6xl-lemonade-dflash2-compat-2026-08-23.json`](docs/results/qwen3.8-q6xl-lemonade-dflash2-compat-2026-08-23.json).

DS4 exposes an OpenAI-compatible API on loopback. Inspect the loaded model and
send a non-thinking chat request with:

```bash
curl --fail http://127.0.0.1:8000/v1/models | jq

curl --fail-with-body \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {"role": "user", "content": "Explain why a persistent KV cache helps repeated long prompts."}
    ],
    "reasoning_effort": "none",
    "max_tokens": 128,
    "stream": false
  }' | jq '.choices[0].message'

halo-ai stop
```

It keeps the proven 32K context and 2,048-token prefill chunk, adds an 8 GiB
budgeted cache at `/var/cache/halo-ai/ds4-kv`, and rejects cache entries from a
different quantization. Cache files survive `stop`, restart, and container
recreation; they do not reduce the memory required by an active context. The
first prompt must still be processed in full before a reusable entry exists.

## Seamless speech translation

The speech service follows AMD's current gfx1151 ROCm playbook and is independent
of Lemonade and the LLM runtimes. Build its cached image layers, download only
the pinned safetensors/processor subset, and verify the local bytes:

```bash
halo-ai install speech
halo-ai models download seamless-m4t-v2-large
halo-ai models verify seamless-m4t-v2-large --full
```

The model is stored directly at
`/srv/halo-ai/models/facebook/seamless-m4t-v2-large`; there is no intermediate
`huggingface/` directory. Downloads resume partial files and reuse complete
ones. The 7.1 GB local image retains the AMD ROCm/PyTorch wheels in Podman's
layer cache, while the model remains external and read-only at runtime.

```bash
halo-ai start seamless-m4t-v2-large-speech --switch
halo-ai test seamless-m4t-v2-large-speech
halo-ai status
```

Open `http://127.0.0.1:7860/` for the local Gradio UI. Automation can use
`GET /healthz` and multipart `POST /api/v1/translate` on the same loopback port.
The built-in test sends AMD's hash-pinned sample to Spanish (`spa`) and validates
that the response is a non-empty WAV. `halo-ai stop` includes this service;
normal uninstall still preserves the external model root.

Query the available three-letter language codes and translate a local audio file
to Ukrainian with:

```bash
curl --fail http://127.0.0.1:7860/healthz | jq '.target_languages'

curl --fail-with-body \
  http://127.0.0.1:7860/api/v1/translate \
  -F 'audio=@/path/to/input.wav' \
  -F 'target_lang=ukr' \
  --output translated-uk.wav

file translated-uk.wav
halo-ai stop
```

The service accepts common audio formats supported by libsndfile, resamples the
input to 16 kHz when necessary, and returns a 16 kHz PCM WAV. Its response
headers include `X-Halo-AI-Inference-Seconds` and
`X-Halo-AI-Target-Language`; add `--dump-header -` to the translation command
when those values are useful. The server derives its language list from the
loaded checkpoint and exposes all 36 languages supported for both speech input
and speech output; the API and Gradio UI use the same list.

## Long-context benchmark

LongBench-v2 support uses a pinned dataset revision and SHA-256, the official
zero-shot prompt, exact chat-template-aware token counts from the active
llama.cpp backend, and resumable JSONL output. Start with the bounded canary:

```bash
halo-ai start qwen3.6-35b-a3b-q8xl-128k-lemonade
halo-ai bench longbench-v2 download
halo-ai bench longbench-v2 run qwen3.6-35b-a3b-q8xl-128k-lemonade
```

Then run the native-fit full subset. Inputs beyond 128K are reported as skipped,
not silently truncated:

```bash
halo-ai bench longbench-v2 run qwen3.6-35b-a3b-q8xl-128k-lemonade --suite full
```

For comparison with the upstream runner's middle-truncation policy, use a
separate output identity:

```bash
halo-ai bench longbench-v2 run qwen3.6-35b-a3b-q8xl-128k-lemonade \
  --suite full --overflow middle
```

Reasoning-policy comparisons use separate resumable outputs and preserve the
historical non-thinking default when the option is omitted:

```bash
halo-ai bench longbench-v2 run qwen3.8fp4 --sample-id SAMPLE \
  --max-tokens 1024 --reasoning-effort default
halo-ai bench longbench-v2 run qwen3.8fp4 --sample-id SAMPLE \
  --max-tokens 1024 --reasoning-effort medium
halo-ai bench longbench-v2 run qwen3.8fp4 --sample-id SAMPLE \
  --max-tokens 1024 --reasoning-effort xhigh
```

Every sample is flushed to disk, so rerunning the identical command resumes.
The score always reports completed, skipped, error, and truncated counts.

For the narrower Qwen3.8 final-answer reliability question, run the paired,
interleaved fixed suite. Two repetitions produce 26 matched calls per arm and
checkpoint atomically after every response:

```bash
halo-ai start qwen3.8fp4
halo-ai bench reasoning-reliability qwen3.8fp4 --repetitions 2 --max-tokens 1024
```

This separately reports API/protocol errors, empty content with `finish_reason`
`stop`, output-budget exhaustion, final-answer delivery, and validator results.
Its Wilson intervals and paired ratios support only a bounded claim for the
recorded host, runtime, suite, sampling policy, and output budget.

## Verified test findings

These measurements were collected on 2026-08-09 PDT (2026-08-10 UTC). They are
an optimization baseline for this exact host, model quantization, and runtime—not
a general model ranking. Memory values are binary GiB derived from amdgpu sysfs;
"GTT used" is dynamic GPU-addressable memory and does not include all host or
fixed-VRAM use.

### Host and runtime baseline

| Item | Verified value | Notes |
| --- | ---: | --- |
| Physical unified memory | 128 GiB | Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`) |
| Linux `MemTotal` | 123.5 GiB | Corrected small BIOS UMA/fixed-VRAM allocation |
| Fixed VRAM | 2 GiB | Leaves most physical memory CPU-visible; suitable for the current Linux topology |
| GTT aperture | 118 GiB | `amdgpu.gttsize=120832`, exact sysfs value `126701535232` bytes |
| TTM page limit | 30,932,992 | 4 KiB pages; exact 118 GiB pair |
| IOMMU/NPU profile | GPU | `amd_iommu=off`; NPU intentionally inactive |
| Qwen server | Lemonade + ROCm | Package `b10334`, active llama.cpp `b10333`, fingerprint `b10333-08659901c` |
| Lemonade image | `sha256:d0d9cc9ead310578d1797bd58b7c583dc007a87bdb04162dde58e0e05ce51794` | Full digest is retained in trial and benchmark manifests |
| Standalone llama.cpp | build `b10335-74ce15741` | `rocm-7.14` image, digest `sha256:32d25e6f7608e1d221b71f51389c883afc655b9a3add9f7a787453dca288117b` |
| ds4 image | manifest `sha256:f9dd84e76c2fbdd3f99b2e0490c40b899586dd047b0bfa0030744cbd58e1df89` | Local ID `52763e9d…493113a`; project-built `b0001`, Antirez `84cc882` ROCm DSpark fix, ROCm `7.15.0a20260728` |
| Speech image | manifest `sha256:0a21384bf020782d8c75df78338bbc8a23f260c3604e5b023eee7a3381d9361b` | Local image ID `532f5f3d…23633`, 7.1 GB; PyTorch 2.12.0 + ROCm 7.14, Transformers 4.57.1, Gradio 6.16.0 |
| Automated source tests | 98 passed | Python unit tests inside the read-only, network-disabled Podman test container, plus shell smoke/install assertions |
| Runtime cleanup | Passed | `halo-ai stop` returned GTT use from about 38 GiB to about 0.1 GiB |

### End-to-end model/profile matrix

| Profile | Context | Feature | GTT used | CPU available | PP tok/s | TPS | Sample time | Result and optimization note |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `qwen3.6-27b-q8xl-lemonade` | 32K | Text | 33.1 GiB | 81.1 GiB | 211.2 | 6.79 | 50.1 s | Pass; no `--mmproj` after isolation fix |
| `qwen3.6-35b-a3b-q8xl-lemonade` | 32K | Text/MoE | 36.2 GiB | 79.8 GiB | 556.6 | 46.70 | 18.7 s | Pass; fastest plain baseline |
| `qwen3.6-27b-q8xl-mtp-lemonade` | 32K | MTP | 34.1 GiB | 80.0 GiB | 199.7 | 11.39 | 52.4 s | Pass; MTP improved decode, not prefill; no projector |
| `qwen3.6-35b-a3b-q8xl-mtp-lemonade` | 32K | MTP/MoE | 37.3 GiB | 78.6 GiB | 525.6 | 52.40 | 19.8 s | Pass; highest Qwen decode TPS in this sample |
| `qwen3.6-27b-q8xl-vision-lemonade` | 32K | Vision | 35.1 GiB | 79.1 GiB | 214.5 | 6.85 | 49.3 s | Text sample passed; separate red-image canary validates F32 projector |
| `qwen3.6-35b-a3b-q8xl-65k-lemonade` | 65K | Text/MoE | 36.8 GiB | 79.3 GiB | 553.5 | 46.46 | 18.9 s | Pass; backend confirmed `--ctx-size 65536` |
| `qwen3.6-35b-a3b-q8xl-128k-lemonade` | 128K | Text/MoE | 38.0 GiB | 77.8 GiB | 557.2 | 46.68 | 18.7 s | Pass; backend confirmed `--ctx-size 131072` |
| `qwen3.6-27b-q8xl-128k-lemonade` | 128K | Text | 39.1 GiB | 75.1 GiB | 215.5 | 6.85 | 49.1 s | Pass; no implicit projector |
| `qwen3.6-35b-a3b-q8xl-128k-mtp-lemonade` | 128K | MTP/MoE | 39.4 GiB | 76.4 GiB | 545.8 | 49.01 | 19.1 s | Pass; MTP decode improvement remains modest on 8 output tokens |
| `qwen3.6-27b-q8xl-128k-vision-lemonade` | 128K | Vision | 41.1 GiB | 73.0 GiB | 218.2 | 6.85 | 48.5 s | Text sample passed; separate red-image canary validates projector |
| `qwen3.6-27b-q8xl-vision-llamacpp` | 32K | Vision | 35.2 GiB | 78.2 GiB | 201.2 | 7.00 | 52.5 s | Pass on standalone build b10335; red-image canary validates projector |
| `qwen3.6-27b-q8xl-mtp-llamacpp` | 32K | MTP | 34.2 GiB | 79.0 GiB | 191.0 | 11.33 | 54.8 s | Pass; smoke metrics reported 6/6 draft tokens accepted |
| `qwen3.6-35b-a3b-q8xl-mtp-llamacpp` | 32K | MTP/MoE | 37.4 GiB | 78.0 GiB | 518.5 | 56.16 | 20.1 s | Pass; fastest measured decode row |
| `deepseek-v4-flash-0731-iq3xxs-llamacpp` | 32K | DeepSeek | 97.3 GiB | 19.1 GiB | 56.78 | 10.21 | 183.9 s | Pass; native DeepSeek reasoning field accepted by its smoke policy |
| `deepseek-v4-flash-0731-iq3xxs-dspark-llamacpp` | 32K | DSpark | 107.5 GiB | 9.2 GiB | 36.43 | 15.31 | 283.7 s | Pass; 24/27 drafts accepted, but worse overall for long input/short output |
| `ds4-deepseek-v4-flash-hybrid` | 32K | ds4/ROCm | 107.6 GiB | 8.9 GiB | 97.73 | 12.56 | 107.2 s | Pass, but tight; PP/TPS parsed from ds4 server log |
| `ds4-deepseek-v4-flash-hybrid-kv` cold | 32K | ds4 + disk KV | 107.6 GiB | 9.2 GiB | 97.37 | 12.43 | 107.7 s | Pass; stored a 10,240-prefix-token entry (157.35 MiB) |
| `ds4-deepseek-v4-flash-hybrid-kv` restored | 32K | ds4 + disk KV | 107.6 GiB | 9.8 GiB | 25.01* | 12.83 | 3.7 s | Pass after container recreation; 10,240 tokens restored in 92.7 ms, only 38-token suffix prefetched |
| `ds4-deepseek-v4-flash-hybrid-dspark-16k` | 16K | ds4 + DSpark | — | — | 29.56* | 8.62 | 46.4 s decode | Pass on fixed ROCm runtime; 246/324 draft tokens accepted (75.93%), zero verifier/runtime errors; enabled but experimental |
| `ds4-deepseek-v4-flash-hybrid-dspark-128k` | 128K | ds4 + DSpark + disk KV | 104.38 GiB | 12.28 GiB | 29.91* | 8.72 | 45.9 s decode | Allocation/smoke pass; 1.78 GiB KV, 246/324 drafts accepted, zero errors; conservative work profile |
| `ds4-deepseek-v4-flash-hybrid-dspark-256k` | 256K | ds4 + DSpark + disk KV | 106.30 GiB | 10.36 GiB | 29.59* | 8.81 | 45.4 s decode | Allocation/smoke pass; 3.46 GiB KV, 246/324 drafts accepted, zero errors; high-context work profile |
| `ds4-deepseek-v4-flash-hybrid-dspark-384k-think-max` | 384K | Think Max + DSpark + disk KV | 108.23 GiB | 8.40 GiB | 29.01* | 8.67 | 46.1 s decode | Think Max response passed; 5.14 GiB KV, 353/436 drafts accepted across both probes, zero errors; opt-in tighter profile |

Notes:

- Qwen text and vision identities use separate container directories while
  bind-mounting the same main GGUF. This avoids model duplication and prevents
  Lemonade from silently attaching `mmproj` to text profiles.
- Every Qwen row used one slot, Flash Attention, F16 K/V, batch 2048, ubatch
  512, no mmap, and the pinned non-thinking-default Jinja template. MTP rows
  used `--spec-type draft-mtp --spec-draft-n-max 2`; vision and MTP remain
  mutually exclusive.
- Lemonade reported installed ROCm package `b10334` while the spawned server
  path was `llama-b10333`. Both values are recorded because a package/channel
  resolution and the binary actually serving requests are not interchangeable.
- The 118 GiB aperture is ample for the tested Qwen 128K profiles. DeepSeek ds4
  is the limiting workload: its observed 107.6 GiB GTT use leaves little shared
  aperture and host-memory headroom despite passing the smoke test.
- GTT and CPU columns are the post-request readings from the fixed matrix
  sample. The earlier 82K/115K canary below shows that longer prefill changes
  latency far more than it changes the preallocated 128K KV memory footprint.
- PP and TPS use the same LongBench-v2 sample `66f37eb9821e116aacb2d295`:
  10,326 rendered Qwen tokens, 10,254 with standalone DeepSeek, and 10,278 with
  ds4. Qwen generated 8 tokens, standalone DeepSeek 33, and ds4 26. The original
  16 LLM profiles completed with no overflow, truncation, or runtime error; the
  later DS4 disk-KV profile repeated the same fixture cold and restored.
  Every Qwen and standalone DeepSeek profile predicted D and ds4 predicted C for
  ground-truth B, so this single case is a systems benchmark, not a quality
  score.
- Standalone Qwen and DeepSeek rows use the refreshed `b10335` image. DSpark
  numbers are a cold run after restarting the container; a retained warm-cache
  artifact completed in 19.1 seconds and is intentionally excluded from the
  matrix. DSpark improves decode but its extra 10.1 GiB companion and slower
  prefill make it a poor default for long-prompt, short-answer work.
- The restored DS4 row's PP value applies only to the 38-token uncached suffix;
  comparing it directly to full-prompt PP is misleading. The meaningful result
  is end-to-end latency falling from 107.7 to 3.7 seconds with identical output.

### Seamless speech service findings

| Item | Verified result | Notes |
| --- | ---: | --- |
| Model subset | 9,258,124,450 bytes (8.62 GiB) | All 12 selected files passed exact size, format sanity, and full SHA-256 verification |
| Pinned revision | `5f8cc790b19fc3f67a61c105133b20b34e3dcb76` | Mutable repository head is not used |
| Model load/start | ~8 seconds on final cached image | Two safetensors shards; local/offline load only |
| GTT after inference | 4.95 GiB | Much smaller than the DeepSeek/Qwen services; still mutually exclusive by lifecycle policy |
| CPU `MemAvailable` after inference | 109.5 GiB | 118 GiB aperture remained compatible with the speech workload |
| AMD WAV to Spanish WAV | 7.04 s; 58,924 output bytes | Valid RIFF/WAVE from the built-in multipart smoke test |
| AMD WAV to Ukrainian WAV | 7.22 s; 65,964 output bytes | Valid 16 kHz mono PCM WAV; `ukr` exercised after enabling all 36 bidirectional speech languages |
| UI/API exposure | Loopback only, port 7860 | UI and health/API checks passed; no Gradio public sharing |

Gradio 6.16.0 is intentional: it is the newest release whose published
Hugging Face Hub dependency overlaps AMD's pinned Transformers 4.57.1. Current
Gradio 6.18+ requires Hub 1.x, while Transformers 4.57.1 requires Hub below 1.0.
The build fails closed rather than forcing an incompatible environment.

### LongBench-v2 128K canary

Dataset revision `2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`, SHA-256
`15d61c22…04c7fe2`, was tested with the 35B-A3B 128K profile and the official
zero-shot prompt/128-token output cap. The six cases are a deterministic
difficulty × length coverage canary.

| Difficulty | Length | Rendered input | Time | Prediction | Outcome |
| --- | --- | ---: | ---: | --- | --- |
| Easy | Long | 675,236 tokens | — | — | Correctly skipped: exceeds 130,944-token input budget |
| Easy | Medium | 82,325 tokens | 228.6 s | A (answer D) | Incorrect |
| Easy | Short | 29,700 tokens | 61.8 s | C (answer C) | Correct |
| Hard | Long | 661,419 tokens | — | — | Correctly skipped: exceeds 130,944-token input budget |
| Hard | Medium | 115,368 tokens | 389.2 s | Unparsed (answer B) | 128-token output cap reached before required answer form |
| Hard | Short | 15,810 tokens | 34.0 s | Unparsed (answer C) | 128-token output cap reached before required answer form |

| Canary aggregate | Value |
| --- | ---: |
| Selected / completed / overflow-skipped | 6 / 4 / 2 |
| Correct / compatible-subset accuracy | 1 / 25.0% |
| Errors / truncated inputs | 0 / 0 |
| Observed prompt processing | ~300–510 tokens/s, decreasing with longer KV history |
| GTT during completed cases | ~38.0 GiB |
| CPU `MemAvailable` during completed cases | ~78 GiB |
| Resume verification | Identical rerun completed in 1.7 s with no repeated inference |

The 25% figure is only a four-sample canary result. It is not statistically
meaningful and must not be compared with the official 503-sample leaderboard.
The two unparsed responses show a useful future optimization target: compare the
official direct-answer cap with a separately labeled larger output budget or a
constrained-answer experiment without overwriting the official-policy run. The
full native-fit and explicit middle-truncation runs are implemented but have not
yet been executed; their deterministic output names allow future Qwen, MTP, and
newer-model comparisons to resume safely.

## Uninstall

```bash
sudo ./uninstall.sh --run-user "$USER" --dry-run
sudo ./uninstall.sh --run-user "$USER"
```

The external model root is never removed. Unlabeled Podman objects and images
are also preserved by default.

## Upstream references

- [AMD Lemonade getting started](https://developer.amd.com/playbooks/lemonade-getting-started/)
- [AMD DeepSeek V4 Flash with ds4](https://developer.amd.com/playbooks/deepseek-v4-flash-ds4/)
- [AMD real-time speech-to-speech translation](https://developer.amd.com/playbooks/speech2speech-translation/)
- [Meta Seamless M4T v2 Large](https://huggingface.co/facebook/seamless-m4t-v2-large)
