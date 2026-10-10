# halo-ai reference manual

[Project README](../README.md)

`halo-ai` acquires verified model artifacts and manages local inference services
on AMD Strix Halo with rootless Podman. This manual describes the checked-in
configuration and supported workflows. The installed catalog and configuration
determine what runs on your host; redeploy checkout changes with `./reload.sh`.

For a first setup, follow **[Installation](installation.md) →
[Profiles](profiles.md) → [Operations](operations.md)**.

## Setup and operation

| Chapter | Contents |
| --- | --- |
| [Installation](installation.md) | Prerequisites, deployment, storage, recovery, and uninstall |
| [Host configuration](host-configuration.md) | Memory topology, GPU/NPU boot profiles, snapshots, and rollback |
| [Profiles and artifacts](profiles.md) | Model/runtime selection, aliases, acquisition, and verification |
| [Operations](operations.md) | Lifecycle commands, client endpoints, caches, and troubleshooting |

## Services

| Chapter | Contents |
| --- | --- |
| [Halogen Flash-Next](halogen-flash.md) | Checkpoint selection, serving defaults, vision, NPU setup, and limitations |
| [Qwen runtimes](qwen-profiles.md) | Lemonade, standalone llama.cpp, DFlash2, ROCmFP4/FP8, and OpenCode |
| [DeepSeek / DS4](ds4.md) | Context profiles, DSpark, persistent KV cache, and API usage |
| [Speech translation](speech.md) | Model setup, web UI, language codes, and translation API |

## Technical reference

| Chapter | Contents |
| --- | --- |
| [Architecture and configuration](halo-ai.md) | Component boundaries, configuration precedence, storage, and artifact manifest |
| [Benchmarks and qualification](benchmarking.md) | Reproducible tests, measurement policy, and evaluation limits |
| [Flash-Next GGUF qualification](qwen3.8-flash-test-plan.md) | MTP controls, tuning procedure, and promotion criteria |
| [Development](development.md) | Checkout validation, repository layout, and documentation maintenance |

Recorded measurements live in [results](results/). Superseded implementation
notes and experiments are preserved in the [evidence archive](archive/README.md).
The [development backlog](../TODO.md) tracks proposed work separately from this manual.
