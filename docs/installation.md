# Installation and host setup

[Documentation index](README.md) · [Project README](../README.md)

Run these commands from the repository root. Host installation deploys the CLI,
configuration, and supporting files. Model acquisition and runtime installation
are separate steps in [Profiles](profiles.md).

## Prerequisites

The reference system is CachyOS on AMD Strix Halo (`gfx1151`) with 128 GiB of
unified memory. The host needs Python 3, Bash, rootless Podman, and
operator access to `/dev/kfd` and `/dev/dri`. GPU libraries come from the runtime
containers; a host ROCm installation is not required.

The installer uses the existing Snapper `root` configuration for pre/post
snapshots. If that configuration is unavailable, repair it or explicitly choose
`--no-snapshot`. See the [snapshot policy](host-configuration.md#snapshot-coverage)
and [host preflight](host-configuration.md#host-preflight) for the full requirements.

Large profiles also depend on the firmware memory split and GTT aperture, not
just installed RAM. Review [host-memory topology](host-configuration.md#host-memory-topology)
before loading them.

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

## Check the installed CLI

Run lifecycle commands as the configured non-root operator:

```bash
halo-ai doctor
halo-ai profiles list
```

Resolve preflight failures before starting a service, then follow
[Choose and acquire a profile](profiles.md#choose-and-acquire-a-profile).

## Configuration and storage

| Concern | Installed location |
| --- | --- |
| CLI and releases | `/usr/local/bin/halo-ai`, `/opt/halo-ai/releases` |
| Configuration | `/etc/opt/halo-ai/config.env` |
| Model catalog and request presets | `/etc/opt/halo-ai/models.d`, `/etc/opt/halo-ai/request-presets.d` |
| External model files | `/srv/halo-ai/models` |
| Runtime state | `/var/opt/halo-ai/state` |
| Runtime caches | `/var/cache/halo-ai` and named Podman volumes |

The checked-in [configuration template](../config/halo-ai.env.example) documents
available settings. See the [configuration reference](halo-ai.md#configuration)
for details.

## GPU memory profile

The following setting is for the documented 128 GiB reference host. The GPU host
profile disables IOMMU; use the [Halogen NPU host instructions](halogen-flash.md#npu-and-host-constraints)
when enabling NPU services.

To stage the reference 118 GiB shared-memory ceiling with a backed-up Limine edit:

```bash
halo-ai host-profile set gpu --gtt-gib 118 --dry-run
sudo halo-ai host-profile set gpu --gtt-gib 118 --yes
```

Reboot manually and require `halo-ai host-profile status` to report both
`running_gtt_gib` and `persistent_gtt_gib` as `118` before long-context tests.

## Installer recovery

Installation stages are `directories`, `release`, `config`, `links`, and `verify`.
A rerun with the same source and operator verifies completed stages before
resuming. Drift requires the reported `--repair-stage STAGE`; different source
content requires `--restart-journal` to archive the prior attempt. The
[reload helper](operations.md#redeploy-checkout-changes) supplies that option
for routine redeployment. Uninstalling first is unnecessary.

A corrected installer can recover a failed attempt against its already verified,
pinned release when the journal and release hash agree. If the application
verification itself fails, deploy the corrected source with a new journal.
Each attempt gets its own Snapper pre/post pair, including failures.

## Uninstall

```bash
sudo ./uninstall.sh --run-user "$USER" --dry-run
sudo ./uninstall.sh --run-user "$USER"
```

The command prints its plan and requires the exact confirmation `uninstall halo-ai`.
For reviewed automation, `--yes` skips that prompt. It removes the installed
application, configuration, state, caches, journal, and labeled project Podman
containers/volumes. The external model root is never removed. Unlabeled objects,
this checkout, and images are preserved by default.

Use `--keep-podman`, `--keep-config`, `--keep-state`, or `--keep-cache` for partial
removal. `--remove-images` applies only to managed-labeled images and cannot be
combined with `--keep-podman`. The uninstaller refuses removal paths that overlap
model storage or contain a mountpoint. See [snapshot coverage](host-configuration.md#snapshot-coverage)
for the difference between root-covered files and separate caches/models.
