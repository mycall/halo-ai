"""Guarded Limine IOMMU host-profile management."""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


LIMINE_DEFAULTS = Path("/etc/default/limine")
BOOT_ROOT = Path("/boot")
STATE_ROOT = Path("/var/opt/halo-ai/state/host-profiles")
BACKUP_ROOT = STATE_ROOT / "backups"
LOCK_PATH = Path("/run/halo-ai-host-profile.lock")
GTT_SETTINGS = {
    112: (114688, 29360128),
    116: (118784, 30408704),
    118: (120832, 30932992),
}


class HostProfileError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_line_from_defaults(text: str) -> tuple[str, re.Match[str]]:
    matches = list(
        re.finditer(
            r'(?m)^(?P<prefix>[ \t]*KERNEL_CMDLINE\[default\]\+=[ \t]*)(?P<quote>["\'])(?P<cmdline>.*)(?P=quote)[ \t]*$',
            text,
        )
    )
    if len(matches) != 1:
        raise HostProfileError("expected exactly one KERNEL_CMDLINE[default]+= assignment")
    return matches[0].group("cmdline"), matches[0]


def classify(cmdline: str) -> str:
    tokens = cmdline.split()
    amd = [item for item in tokens if item.startswith("amd_iommu=")]
    generic = [item for item in tokens if item.startswith("iommu=")]
    if len(amd) > 1 or len(generic) > 1:
        return "invalid-duplicate"
    if amd == ["amd_iommu=off"] and not generic:
        return "gpu"
    if not amd and not generic:
        return "npu"
    return "custom"


def gtt_from_cmdline(cmdline: str) -> int | None:
    tokens = cmdline.split()
    gtt = [item for item in tokens if item.startswith("amdgpu.gttsize=")]
    pages = [item for item in tokens if item.startswith("ttm.pages_limit=")]
    if len(gtt) > 1 or len(pages) > 1:
        raise HostProfileError("duplicate GTT/TTM arguments; refusing ambiguous edit")
    if not gtt and not pages:
        return None
    if len(gtt) != 1 or len(pages) != 1:
        raise HostProfileError("amdgpu.gttsize and ttm.pages_limit must be configured together")
    try:
        values = (int(gtt[0].split("=", 1)[1]), int(pages[0].split("=", 1)[1]))
    except ValueError as exc:
        raise HostProfileError("GTT/TTM arguments must be decimal integers") from exc
    matches = [gib for gib, setting in GTT_SETTINGS.items() if setting == values]
    if len(matches) != 1:
        raise HostProfileError(f"unsupported GTT/TTM setting: {values[0]} MiB/{values[1]} pages")
    return matches[0]


def transform(cmdline: str, profile: str, gtt_gib: int | None = None) -> str:
    tokens = cmdline.split()
    amd = [item for item in tokens if item.startswith("amd_iommu=")]
    generic = [item for item in tokens if item.startswith("iommu=")]
    if len(amd) > 1 or len(generic) > 1:
        raise HostProfileError("duplicate IOMMU arguments; refusing ambiguous edit")
    if amd and amd[0] != "amd_iommu=off":
        raise HostProfileError(f"unsupported AMD IOMMU token: {amd[0]}")
    if generic and generic[0] != "iommu=pt":
        raise HostProfileError(f"unsupported generic IOMMU token: {generic[0]}")
    if gtt_gib is not None and gtt_gib not in GTT_SETTINGS:
        raise HostProfileError(f"unsupported GTT aperture: {gtt_gib} GiB")
    if gtt_gib is not None:
        # Refuse to repair or normalize a partial/custom pair while performing
        # an aperture change; it needs explicit operator review instead.
        gtt_from_cmdline(cmdline)
    gtt_tokens = [
        item for item in tokens
        if item.startswith("amdgpu.gttsize=") or item.startswith("ttm.pages_limit=")
    ] if gtt_gib is not None else []
    filtered = [item for item in tokens if item not in amd and item not in generic and item not in gtt_tokens]
    if profile == "gpu":
        filtered.append("amd_iommu=off")
    elif profile != "npu":
        raise HostProfileError(f"unknown host profile: {profile}")
    if gtt_gib is not None:
        gtt_mib, page_limit = GTT_SETTINGS[gtt_gib]
        filtered.extend([f"amdgpu.gttsize={gtt_mib}", f"ttm.pages_limit={page_limit}"])
    return " ".join(filtered)


def render_defaults(text: str, profile: str, gtt_gib: int | None = None) -> tuple[str, str, str]:
    old, match = command_line_from_defaults(text)
    new = transform(old, profile, gtt_gib)
    replacement = f"{match.group('prefix')}{match.group('quote')}{new}{match.group('quote')}"
    return text[: match.start()] + replacement + text[match.end() :], old, new


def atomic_write(path: Path, data: str, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def boot_files() -> list[Path]:
    if not BOOT_ROOT.is_dir():
        raise HostProfileError("/boot is unavailable")
    return [item for item in BOOT_ROOT.rglob("*") if item.is_file() and not item.is_symlink()]


def create_backup(identifier: str) -> dict[str, Any]:
    destination = BACKUP_ROOT / identifier
    if destination.exists():
        raise HostProfileError(f"backup already exists: {identifier}")
    destination.mkdir(parents=True, mode=0o700)
    config_backup = destination / "etc-default-limine"
    config_record = None
    if LIMINE_DEFAULTS.is_file():
        shutil.copy2(LIMINE_DEFAULTS, config_backup)
        config_record = {"bytes": config_backup.stat().st_size, "sha256": sha256(config_backup)}
    files = boot_files()
    total = sum(item.stat().st_size for item in files)
    free = shutil.disk_usage(destination).free
    if free < total + 1024**3:
        shutil.rmtree(destination)
        raise HostProfileError(f"insufficient backup capacity: need {total + 1024**3} bytes, have {free}")
    boot_destination = destination / "boot"
    manifest_files = []
    for source in files:
        relative = source.relative_to(BOOT_ROOT)
        target = boot_destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest_files.append({"path": str(relative), "bytes": target.stat().st_size, "sha256": sha256(target)})
    manifest = {
        "schema_version": 1,
        "id": identifier,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "limine_defaults": config_record,
        "boot_files": manifest_files,
    }
    atomic_write(destination / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n", 0o600)
    return manifest


def verify_backup(identifier: str) -> tuple[Path, dict[str, Any]]:
    directory = BACKUP_ROOT / identifier
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HostProfileError(f"cannot read backup manifest {identifier}: {exc}") from exc
    config = directory / "etc-default-limine"
    expected = manifest.get("limine_defaults")
    if not isinstance(expected, dict):
        raise HostProfileError(f"backup {identifier} predates a Limine defaults file and cannot be rolled back")
    if config.stat().st_size != expected.get("bytes") or sha256(config) != expected.get("sha256"):
        raise HostProfileError(f"Limine backup checksum failed: {identifier}")
    for item in manifest.get("boot_files", []):
        path = directory / "boot" / item["path"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise HostProfileError(f"boot backup checksum failed: {item['path']}")
    return directory, manifest


def snapper_pre(description: str) -> str:
    check = subprocess.run(["snapper", "-c", "root", "get-config"], text=True, capture_output=True)
    if check.returncode != 0:
        raise HostProfileError(f"Snapper root config is unavailable: {(check.stderr or check.stdout).strip()}")
    try:
        result = subprocess.run(
            ["snapper", "-c", "root", "create", "--type", "pre", "--print-number", "--description", description,
             "--cleanup-algorithm", "number", "--userdata", "important=yes,halo_ai=host_profile"],
            check=True, text=True, capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise HostProfileError(f"could not create Snapper pre-snapshot: {(exc.stderr or exc.stdout).strip()}") from exc
    return result.stdout.strip()


def snapper_post(pre: str, description: str, outcome: str) -> str:
    try:
        result = subprocess.run(
            ["snapper", "-c", "root", "create", "--type", "post", "--pre-number", pre, "--print-number",
             "--cleanup-algorithm", "number", "--description", description,
             "--userdata", f"important=yes,halo_ai=host_profile,outcome={outcome}"],
            check=True, text=True, capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise HostProfileError(f"could not create Snapper post-snapshot for {pre}: {(exc.stderr or exc.stdout).strip()}") from exc
    return result.stdout.strip()


def persistent_status() -> dict[str, Any]:
    running = Path("/proc/cmdline").read_text().strip()
    persistent = ""
    error = ""
    try:
        persistent, _ = command_line_from_defaults(LIMINE_DEFAULTS.read_text(encoding="utf-8"))
    except (OSError, HostProfileError) as exc:
        error = str(exc)
    running_profile = classify(running)
    persistent_profile = classify(persistent) if persistent else "unavailable"
    driver = Path("/sys/bus/pci/drivers/amdxdna")
    amdxdna_bound = driver.is_dir() and any(
        item.is_symlink() and re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", item.name)
        for item in driver.iterdir()
    )
    try:
        running_gtt = gtt_from_cmdline(running)
    except HostProfileError:
        running_gtt = None
    try:
        persistent_gtt = gtt_from_cmdline(persistent) if persistent else None
    except HostProfileError as exc:
        persistent_gtt = None
        error = f"{error}; {exc}".strip("; ")
    return {
        "running_profile": running_profile,
        "persistent_profile": persistent_profile,
        "reboot_pending": (
            (running_profile != persistent_profile and persistent_profile in {"gpu", "npu"})
            or (persistent_gtt is not None and running_gtt != persistent_gtt)
        ),
        "running_gtt_gib": running_gtt,
        "persistent_gtt_gib": persistent_gtt,
        "running_cmdline": running,
        "persistent_cmdline": persistent,
        "persistent_error": error,
        "iommu_groups_present": any(Path("/sys/kernel/iommu_groups").glob("*")),
        "amdxdna_bound": amdxdna_bound,
        "npu_device": Path("/dev/accel/accel0").exists(),
    }


def require_root() -> None:
    if os.geteuid() != 0:
        raise HostProfileError("host-profile mutations require sudo/root")


@contextlib.contextmanager
def mutation_lock():
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise HostProfileError("another host-profile mutation is active") from exc
        yield


def confirm(words: str, assume_yes: bool) -> None:
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise HostProfileError("interactive confirmation unavailable; inspect --dry-run, then use --yes")
    if input(f'Type "{words}" to continue: ') != words:
        raise HostProfileError("confirmation did not match")


def mutate(profile: str, *, gtt_gib: int | None, dry_run: bool, assume_yes: bool, no_snapshot: bool) -> int:
    if not dry_run:
        require_root()
    text = LIMINE_DEFAULTS.read_text(encoding="utf-8")
    rendered, old, new = render_defaults(text, profile, gtt_gib)
    print(f"persistent cmdline before: {old}")
    print(f"persistent cmdline after:  {new}")
    if old == new:
        print(f"Host profile {profile} is already persistent.")
        return 0
    if profile == "gpu" and Path("/dev/accel/accel0").exists():
        raise HostProfileError("refusing GPU profile while the NPU device is active; stop NPU workloads first")
    if profile == "npu":
        print("warning: NPU mode also requires IOMMU enabled in firmware and a matched amdxdna stack", file=sys.stderr)
    if dry_run:
        print("Dry run: no files, snapshots, or boot entries were changed.")
        return 0
    confirm(f"stage halo-ai {profile} profile", assume_yes)
    description = f"halo-ai host-profile {profile}"
    if gtt_gib is not None:
        description += f" gtt-{gtt_gib}"
    identifier = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pre = ""
    outcome = "failed"
    if not no_snapshot:
        pre = snapper_pre(description)
    try:
        manifest = create_backup(identifier)
        mode = LIMINE_DEFAULTS.stat().st_mode & 0o777
        atomic_write(LIMINE_DEFAULTS, rendered, mode)
        subprocess.run(["/usr/bin/limine-mkinitcpio"], check=True)
        outcome = "complete"
        print(f"Backup: {identifier} ({len(manifest['boot_files'])} /boot files verified)")
        print("Change staged. Reboot manually, then run: halo-ai host-profile status")
        return 0
    except BaseException:
        backup = BACKUP_ROOT / identifier / "etc-default-limine"
        if backup.is_file():
            shutil.copy2(backup, LIMINE_DEFAULTS)
            with contextlib.suppress(subprocess.CalledProcessError):
                subprocess.run(["/usr/bin/limine-mkinitcpio"], check=True)
        raise
    finally:
        if pre:
            post = snapper_post(pre, f"{description} {outcome}", outcome)
            print(f"Snapper pair: {pre} -> {post} ({outcome})")


def initialize(*, dry_run: bool, assume_yes: bool, no_snapshot: bool) -> int:
    if not dry_run:
        require_root()
    if LIMINE_DEFAULTS.exists():
        command_line_from_defaults(LIMINE_DEFAULTS.read_text(encoding="utf-8"))
        print(f"Limine defaults already initialized: {LIMINE_DEFAULTS}")
        return 0
    running = Path("/proc/cmdline").read_text().strip()
    proposed = f'ESP_PATH="/boot"\nKERNEL_CMDLINE[default]+="{running}"\nBOOT_ORDER="*, *lts, *fallback, Snapshots"\n'
    print(proposed, end="")
    if dry_run:
        return 0
    confirm("initialize halo-ai Limine profile", assume_yes)
    pre = "" if no_snapshot else snapper_pre("halo-ai host-profile init")
    outcome = "failed"
    identifier = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        manifest = create_backup(identifier)
        atomic_write(LIMINE_DEFAULTS, proposed, 0o644)
        subprocess.run(["/usr/bin/limine-mkinitcpio"], check=True)
        outcome = "complete"
        print(f"Backup: {identifier} ({len(manifest['boot_files'])} /boot files verified)")
        return 0
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            LIMINE_DEFAULTS.unlink()
        raise
    finally:
        if pre:
            print(f"Snapper pair: {pre} -> {snapper_post(pre, f'halo-ai host-profile init {outcome}', outcome)} ({outcome})")


def rollback(identifier: str, *, dry_run: bool, assume_yes: bool, no_snapshot: bool) -> int:
    if not dry_run:
        require_root()
    directory, _manifest = verify_backup(identifier)
    backup = directory / "etc-default-limine"
    old, _ = command_line_from_defaults(LIMINE_DEFAULTS.read_text(encoding="utf-8"))
    restored_text = backup.read_text(encoding="utf-8")
    new, _ = command_line_from_defaults(restored_text)
    print(f"persistent cmdline before: {old}")
    print(f"persistent cmdline restore: {new}")
    if dry_run:
        return 0
    confirm(f"rollback halo-ai host profile {identifier}", assume_yes)
    pre = "" if no_snapshot else snapper_pre(f"halo-ai host-profile rollback {identifier}")
    outcome = "failed"
    try:
        atomic_write(LIMINE_DEFAULTS, restored_text, LIMINE_DEFAULTS.stat().st_mode & 0o777)
        subprocess.run(["/usr/bin/limine-mkinitcpio"], check=True)
        outcome = "complete"
        print("Rollback staged. Reboot manually and verify host-profile status.")
        return 0
    finally:
        if pre:
            print(f"Snapper pair: {pre} -> {snapper_post(pre, f'halo-ai rollback {outcome}', outcome)} ({outcome})")
