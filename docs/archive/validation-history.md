# Historical validation results

> Archived evidence. Settings and host state describe the original trials.
> Use the [reference manual](../README.md) for supported operations.

[Documentation index](../README.md) · [Project README](../../README.md)

These are the host and runtime measurements formerly recorded in the project
README. They preserve the original test conditions and limitations; runtime pins
and defaults may have changed since these runs. Use [Profiles](../profiles.md) for
current selection, [Halogen](../halogen-flash.md) for its current decision, and
[Benchmarks](../benchmarking.md) to reproduce tests.

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
