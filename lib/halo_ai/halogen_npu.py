"""Pinned optional Halogen NPU artifacts and read-only host checks."""
from __future__ import annotations

import json
import resource
from pathlib import Path
import artifact_store
from artifact_store import Artifact
from errors import fail

MANIFEST = json.loads(Path(__file__).with_name('halogen_npu_models.json').read_text())
MODEL_IDS = frozenset(MANIFEST['models'])
LIBRARIES = ('libxrt_coreutil.so.2', 'libxrt_core.so.2', 'libxrt_driver_xdna.so.2')


def artifacts(root: Path, models: list[str]):
    """Include shared device programs without downloading their owner's weights."""
    wanted = {}
    for name in models:
        model = MANIFEST['models'][name]
        owner = model.get('devices', name)
        for item in model['files']:
            if owner == name or not item['path'].startswith('devices/'):
                wanted[name, item['path']] = item
        if owner != name:
            for item in MANIFEST['models'][owner]['files']:
                if item['path'].startswith('devices/'):
                    wanted[owner, item['path']] = item
    for (name, relative), item in sorted(wanted.items()):
        model = MANIFEST['models'][name]
        yield root / name / relative, item, (
            f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{relative}"
        )


def pinned_artifacts(root: Path, models: list[str]) -> tuple[Artifact, ...]:
    result = []
    root = root.resolve()
    for path, item, _ in artifacts(root, models):
        try:
            path.resolve().relative_to(root)
        except ValueError:
            fail(f"NPU artifact path escapes model root: {path}")
        name = path.relative_to(root).parts[0]
        model = MANIFEST['models'][name]
        result.append(Artifact(path, 'npu', item['bytes'], item['sha256'],
                               model['repo'], model['revision'], item['path']))
    return tuple(result)


def verify(root: Path, models: list[str], full: bool = False) -> list[str]:
    return artifact_store.verify(pinned_artifacts(root, models), full=full)


def download(root: Path, models: list[str]) -> None:
    artifact_store.acquire(root, pinned_artifacts(root, models))


def xrt_mounts(directory: Path) -> list[tuple[Path, str]]:
    if not directory.is_absolute():
        raise ValueError('HALOGEN_XRT_LIB_DIR must be an absolute host library directory')
    mounts = []
    for name in LIBRARIES:
        source = (directory / name).resolve(strict=True)
        if not source.is_file():
            raise ValueError(f'XRT library is not a file: {source}')
        # The distro plugin also looks for XRT at its original library path.
        mounts.extend([(source, f'/opt/xilinx/xrt/lib/{name}'), (source, str(directory / name))])
    return mounts


def host_errors(directory: Path, sysfs: Path = Path('/sys'), device: Path = Path('/dev/accel/accel0'), required_memlock: int = 0) -> list[str]:
    errors = []
    hard_limit = resource.getrlimit(resource.RLIMIT_MEMLOCK)[1]
    if hard_limit != resource.RLIM_INFINITY and hard_limit < required_memlock:
        errors.append(f'operator memlock limit is {hard_limit} bytes; NPU profile reserves {required_memlock} bytes; run halogen_memlock_setup.py and use a renewed login')
    if not device.exists():
        errors.append('NPU device missing; select the npu host profile and reboot with BIOS IOMMU enabled')
    try:
        xrt_mounts(directory)
    except (OSError, ValueError) as exc:
        errors.append(f'XRT unavailable: {exc}')
    controls = list(sysfs.glob('class/drm/card*/device/pp_dpm_fclk'))
    if not controls:
        errors.append('GPU fabric clock control unavailable')
    for control in controls:
        try:
            mode = control.with_name('power_dpm_force_performance_level').read_text().strip()
            levels = [line for line in control.read_text().splitlines() if line.strip()]
            held = mode == 'high' or (mode == 'manual' and levels and '*' in levels[-1] and sum('*' in line for line in levels) == 1)
            if not held:
                errors.append(f'GPU fabric clock is not held: {control.parent}; enable halogen-fabric-clock.service')
        except OSError as exc:
            errors.append(str(exc))
    return errors
