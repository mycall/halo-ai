# halo-ai

Run and manage local AI services on AMD Strix Halo with CachyOS and rootless
Podman. `halo-ai` handles model acquisition and verification, runtime setup,
profile switching, health checks, and benchmarking through one CLI.

Models live outside containers in `/srv/halo-ai/models`. Runtime images are
pinned, caches survive restarts, and the lifecycle manager keeps one large
inference runtime active at a time.

## Supported services

- **Qwen:** Halogen Flash-Next, Lemonade, standalone llama.cpp, Strix Vulkan,
  and ROCmFPX profiles for text, vision, and speculative decoding.
- **DeepSeek V4 Flash:** DS4 profiles with optional disk KV caching and DSpark.
- **Speech translation:** Seamless M4T v2 with a local web UI and API.
- **Optional NPU services:** Halogen embeddings and reranking.

Profile support and qualification vary. See the [profile guide](docs/profiles.md)
for aliases, runtime choices, and experimental limitations.

## Getting started

The reference host is a 128 GiB AMD Strix Halo (`gfx1151`) system running
CachyOS. You need Python, Bash, rootless Podman, and GPU device access.
The installer uses an existing Snapper `root` configuration by default.
Check the [installation guide](docs/installation.md) for prerequisites and
host memory setup.

From the repository root, preview and install the host tools:

```bash
sudo ./install.sh --run-user "$USER" --dry-run
sudo ./install.sh --run-user "$USER"
```

Then run the CLI as your configured non-root operator:

```bash
halo-ai doctor
halo-ai profiles list
```

Next, [choose and acquire a profile](docs/profiles.md#choose-and-acquire-a-profile).
That guide walks through downloading its artifacts, installing its runtime,
starting the service, and testing it. Large profiles require the documented
memory configuration and substantial model storage.

## Everyday commands

```bash
halo-ai status
halo-ai start qwen3.8-fn --switch  # Start an acquired profile; switch runtimes if needed
halo-ai test qwen3.8-fn
halo-ai stop                     # Stop all managed runtimes; keep models and caches
```

After editing or updating this checkout, redeploy with `./reload.sh`.
See [operations](docs/operations.md) for client endpoints, restarts, and cache
behavior, or [installation](docs/installation.md#uninstall) for removal.

## Documentation

| Guide | What it covers |
| --- | --- |
| [Installation](docs/installation.md) | Prerequisites, host setup, configuration, and uninstall |
| [Profiles](docs/profiles.md) | Choosing a model/runtime and acquiring its artifacts |
| [Operations](docs/operations.md) | Start, stop, switch, connect clients, and redeploy |
| [Halogen](docs/halogen-flash.md) | Flash-Next setup, vision, NPU services, and limitations |
| [Qwen runtimes](docs/qwen-profiles.md) | DFlash2, MTP, ROCmFP4/FP8, and compatibility |
| [DeepSeek / DS4](docs/ds4.md) | Disk KV cache, DSpark profiles, and API usage |
| [Speech translation](docs/speech.md) | Model setup, web UI, and translation API |
| [Benchmarks](docs/benchmarking.md) | Profile matrix, long-context tests, and reasoning checks |
| [Development](docs/development.md) | Checkout validation and repository layout |

The [reference manual](docs/README.md) also covers host configuration, architecture,
artifact verification, and qualification procedures.

## Development

Run the source, unit, and shell checks in the isolated test container:

```bash
./tests/container.sh
```

The suite uses a read-only checkout and a network-disabled rootless Podman
container. It does not load inference models. See the
[development guide](docs/development.md) for additional checks.

## License

[MIT](LICENSE.md). Models and upstream runtimes have their own licenses.
