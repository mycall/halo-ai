# Development and checkout validation

[Documentation index](README.md) · [Project README](../README.md)

Run all commands from the repository root. Installation and runtime operations are covered in [Installation](installation.md) and [Operations](operations.md).

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
host CLI checks above remain read-only except when an explicit lifecycle verb
such as `install`, `start`, or `stop` is requested.

These commands do not load a model. `models verify --full` additionally hashes
cataloged model files and writes an inventory record. This can read hundreds of
GiB, so it is deliberately not part of the smoke test.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`bin/halo-ai`](../bin/halo-ai) | CLI entry point |
| [`lib/halo_ai/`](../lib/halo_ai/) | Lifecycle, engines, artifacts, and benchmark implementation |
| [`config/`](../config/) | Model catalogs, configuration, request presets, and benchmark fixtures |
| [`tests/`](../tests/) | Unit tests, shell checks, and live qualification runners |
| [`tools/`](../tools/) | Runtime validation and host preparation utilities |
| [`docs/`](README.md) | Reference manual |
| [`docs/results/`](results/) | Recorded qualification artifacts |

For implementation details, see the [architecture and configuration reference](halo-ai.md).

## Documentation maintenance

Maintain the manual by topic. Update the existing explanation when behavior
changes; consolidate or replace superseded guidance instead of appending dated
status sections, session summaries, or upgrade diaries.

Keep procedures, defaults, compatibility constraints, and known limitations in
the relevant chapter. Record measurements with their exact runtime/model identity
and test scope under `docs/results/`, then link to the evidence beside the claim.
Dates belong in evidence metadata and artifact names. A configured context size
or successful smoke test does not establish full-context quality.

Keep proposed work in `TODO.md`. The [archive](archive/README.md) preserves earlier
notes for provenance and is not an ongoing log or a source of current defaults.
Avoid repeating detailed measurements or runtime pins across chapters; link to
the owning service guide, configuration template, or catalog.

When editing the manual, check relative links and heading anchors, compare
commands with CLI help, and run the source suite when changing the
[expected-file manifest](halo-ai.md#expected-file-manifest), which is checked
against the catalogs. Documentation changes alone do not require live model loads.
