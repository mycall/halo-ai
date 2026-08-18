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
halo-ai profiles acquire qwen38-27b-rocmfp4-baseline --dry-run
halo-ai profiles acquire qwen38-27b-rocmfp4-baseline
halo-ai models verify qwen3.8-27b-rocmfp4 --full
halo-ai start qwen38-27b-rocmfp4-baseline --switch
halo-ai test qwen38-27b-rocmfp4-baseline
halo-ai stop qwen38-27b-rocmfp4-baseline
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

Rollback is non-destructive: `halo-ai stop qwen38-27b-rocmfp4-baseline` stops
the service while preserving the verified GGUF and image. Start the previous
profile with `--switch`; remove the local ROCmFPX image separately only when
its cached runtime is no longer wanted.

The same-host Stage 1 qualification, including short/4K/32K cold-prompt
measurements and the direct `llama-bench` control, is recorded in
[`docs/results/qwen38-rocmfp4-baseline-2026-08-16.json`](docs/results/qwen38-rocmfp4-baseline-2026-08-16.json).
Performance claims collected on other q38rocm setups are not used as pass/fail
evidence for this host; these measurements are the baseline for subsequent
optimization.

The separate `qwen38-27b-rocmfp4-mtp` profile reuses the same GGUF and downloads
no model weights. It enables the backend's strict Qwen verifier with the
experimental `n_max=6`, `p_min=0.60` proposal settings. On the same prompts it
improved decode throughput by 29.5% at short context, 59.1% at 4K, and 45.9% at
32K, while increasing TTFT and memory use. Keep it for generation-heavy
experiments; use the baseline for cold long prompts with short answers. The
full results and current conformance caveat are in
[`docs/results/qwen38-rocmfp4-mtp-2026-08-16.json`](docs/results/qwen38-rocmfp4-mtp-2026-08-16.json).

Turn those records into a repeatable gate, and inspect the next no-download
experiment, with:

```bash
halo-ai tune mtp-compare \
  docs/results/qwen38-rocmfp4-baseline-2026-08-16.json \
  docs/results/qwen38-rocmfp4-mtp-2026-08-16.json
halo-ai tune stage3-preflight
halo-ai tune stage3-preflight --full
```

The MTP comparison accepts only matching prompt-token sets and the same model
SHA-256, calculates TTFT/prefill/decode/GTT deltas, and keeps the current
candidate experimental while strict token identity is unresolved. Stage 3
preflight is local-only: it reuses the cached Qwen3.6-35B-A3B model if present,
otherwise says to skip the proxy. It never downloads Q4NX, NPU, vision, or any
other model artifact. `--full` hashes cached files and merges their evidence
into the inventory without deleting earlier model records.

The guarded Stage 3 screen is split around the model switch so both large
models are never resident together:

```bash
halo-ai start qwen38-27b-rocmfp4-baseline --switch
halo-ai tune stage3-capture-target /var/opt/halo-ai/state/stage3-target.json
halo-ai start qwen3.6-35b-a3b-q8xl-lemonade --switch
halo-ai tune stage3-capture-draft \
  /var/opt/halo-ai/state/stage3-target.json \
  /var/opt/halo-ai/state/stage3-proxy.json
halo-ai tune stage3-score /var/opt/halo-ai/state/stage3-proxy.json
halo-ai stop
```

The first short canary found a sharp mode boundary: non-thinking prefixes had
94.29% first-token agreement and a 4.771/6 mean accepted run, while thinking
prefixes had 28.57% and 1.143/6. The general cross-version proxy therefore hit
the early stop gate; no NPU or Q4NX artifact is authorized. See
[`docs/results/qwen38-qwen36-version-mix-canary-2026-08-16.json`](docs/results/qwen38-qwen36-version-mix-canary-2026-08-16.json).

The first Lemonade ROCm start downloads a llama.cpp backend and TheRock runtime.
They are cached in the persistent `halo-lemonade-config` volume, while Hugging
Face downloads use `halo-lemonade-huggingface`. The default
`LEMONADE_LLAMACPP_ROCM_BIN=latest` asks Lemonade to check for the newest stable
ROCm package on each explicit start; already downloaded versions and the large
TheRock runtime are reused. Set a specific `bNNNN` value when reproducibility is
more important than tracking current llama.cpp.

## DS4 disk KV cache

The original DS4 hybrid profile remains the uncached control. Use the separate
cache profile when coding agents or other clients repeatedly send a long shared
prefix:

```bash
halo-ai install ds4
halo-ai start ds4-deepseek-v4-flash-hybrid-kv --switch
halo-ai test ds4-deepseek-v4-flash-hybrid-kv
```

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

Every sample is flushed to disk, so rerunning the identical command resumes.
The score always reports completed, skipped, error, and truncated counts.

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
| ds4 image | `sha256:2ea5b3b28334f08d53307baf79838591e510628d41dacec357de32ffafbac31f` | `kyuz0/strix-halo-ds4-toolbox:rocm-7.14` |
| Speech image | manifest `sha256:0a21384bf020782d8c75df78338bbc8a23f260c3604e5b023eee7a3381d9361b` | Local image ID `532f5f3d…23633`, 7.1 GB; PyTorch 2.12.0 + ROCm 7.14, Transformers 4.57.1, Gradio 6.16.0 |
| Automated source tests | 76 passed | Python unit tests inside the read-only, network-disabled Podman test container, plus shell smoke/install assertions |
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
