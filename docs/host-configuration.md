# Host configuration

[Reference manual](README.md) · [Installation](installation.md)

## Host preflight

Run `halo-ai doctor` before acquiring or loading a large profile. The reference
host is a 128 GiB Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`) running CachyOS.
The host supplies `amdgpu`, `/dev/kfd`, and `/dev/dri`; containers supply GPU
userspace libraries. The non-root operator needs access to the GPU device nodes.

Useful checks when preflight fails:

```bash
uname -r
lscpu
lspci -nnk
ls -l /dev/kfd /dev/dri
id
podman info
free -h
findmnt -T /srv/halo-ai/models
df -h /srv/halo-ai/models "$HOME/.local/share/containers"
```

Check space on both the model filesystem and the rootless Podman store. Stop
other GPU inference jobs before loading a large model. A port or device-access
failure requires fixing the reported conflict; privileged containers and CPU
fallback are not substitutes for the selected GPU runtime.

## Host-memory topology

Unified memory has three separate limits:

1. Firmware divides physical memory between fixed GPU VRAM and CPU-visible RAM.
2. The kernel's GTT/TTM settings limit dynamic GPU-addressable memory.
3. Each runtime decides which memory pools count toward model eligibility.

The reference topology uses 2 GiB fixed VRAM, about 123.5 GiB visible to Linux,
and a 118 GiB GTT aperture. A large fixed-VRAM carveout can starve CPU-side model
loading even when the model fits the GPU pool. GTT is an addressable ceiling,
not extra physical RAM or a reservation of that entire amount.

`doctor` reads amdgpu sysfs, rather than inferring GPU memory from `free` alone.
`halo-ai env` prints the observed hardware snapshot, including VRAM/GTT totals
and use, target aperture, and any pending aperture. Observed values are refreshed
from the host and cannot be overridden by setting the output variables.

| GTT aperture | `amdgpu.gttsize` (MiB) | `ttm.pages_limit` (4 KiB pages) |
| ---: | ---: | ---: |
| 112 GiB | 114688 | 29360128 |
| 116 GiB | 118784 | 30408704 |
| 118 GiB | 120832 | 30932992 |

The configured ceiling is 118 GiB with a 4 GiB OS reserve. Model weights, KV
cache, compute buffers, display use, and other processes all consume memory;
an aperture-minus-file-size calculation does not establish usable context.
`ttm.page_pool_size` is left at the driver default.

Lemonade also needs `enable_dgpu_gtt=true` and the ROCm backend. Inspect
`/api/v1/system-info` when its eligible memory appears limited to fixed VRAM.
See [Lemonade operations](operations.md#lemonade-and-standalone-llamacpp).

## GPU and NPU boot profiles

| Profile | Kernel IOMMU policy | Intended use |
| --- | --- | --- |
| `gpu` | Adds `amd_iommu=off`, removes `iommu=pt` | GPU-only inference; XDNA NPU unavailable |
| `npu` | Removes `amd_iommu=off` and `iommu=pt` | Translated IOMMU for XDNA; firmware IOMMU must also be enabled |

Inspect the running and persistent settings first:

```bash
halo-ai host-profile status
halo-ai host-profile set gpu --gtt-gib 118 --dry-run
```

Apply a reviewed GPU configuration with:

```bash
sudo halo-ai host-profile set gpu --gtt-gib 118 --yes
```

For NPU services, follow [Halogen host setup](halogen-flash.md#npu-and-host-constraints),
which also prepares XRT, fabric clocks, and memlock. Switching IOMMU alone is
insufficient. Do not add `amd_iommu=on`; the NPU profile removes the explicit
disable override.

`host-profile init` is needed if `/etc/default/limine` does not exist. It derives
the configuration from the complete running command line. Preview it with
`--dry-run` before applying it through sudo. `set` accepts `--gtt-gib 112|116|118`
and changes the matching GTT/TTM pair together while preserving unrelated boot
arguments.

Each mutation shows the diff, creates a Snapper pair and checksummed boot-file
backups, and invokes `limine-mkinitcpio`. Do not edit generated
`/boot/limine.conf` directly. Commands never reboot automatically. Reboot manually,
then run `halo-ai host-profile status`; a persistent/running mismatch is still
pending. For the 118 GiB configuration, both `running_gtt_gib` and
`persistent_gtt_gib` must report `118`.

An NPU host also requires initialized IOMMU groups, `amdxdna` binding, and
`/dev/accel/accel0`. GPU mode refuses a switch while an NPU workload is active.
For rollback, select the backup ID reported by the host-profile operation:

```bash
halo-ai host-profile rollback BACKUP_ID --dry-run
sudo halo-ai host-profile rollback BACKUP_ID --yes
```

Rollback also regenerates Limine and requires a manual reboot.

## Snapshot coverage

The reference CachyOS layout separates root `@`, `/srv` (`@srv`), `/var/cache`
(`@cache`), `/home` (`@home`), and VFAT `/boot`. Check your own layout with
`findmnt` and inspect Snapper configuration before relying on that coverage:

```bash
sudo snapper list-configs
sudo snapper -c root get-config
sudo snapper -c root list
```

Host installation and covered configuration changes use a root pre/post pair,
including failed attempts. If the expected `root` configuration is unavailable,
the mutation fails unless `--no-snapshot` is explicitly selected. Read-only CLI
commands, container lifecycle, model downloads, and image pulls do not require
root snapshots. Package transactions use the existing `snap-pac` integration.

Root snapshots do not cover the separate model, cache, home, or boot filesystems.
Host-profile operations therefore also back up boot files under
`/var/opt/halo-ai/state/host-profiles/backups`. Snapshot retention and user-wide
Podman storage policy remain host administration concerns; halo-ai does not
rewrite them.
