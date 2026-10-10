#!/usr/bin/env python3
"""Download/verify the pinned Halogen NPU files; report host readiness."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib/halo_ai'))
import halogen_npu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/srv/halo-ai/models/halogen-npu') / halogen_npu.MANIFEST['artifact_release'])
    parser.add_argument('--models', nargs='+', choices=sorted(halogen_npu.MODEL_IDS), default=['qwen3-embedding-0.6b', 'qwen3-reranker-0.6b'])
    parser.add_argument('--xrt-lib-dir', type=Path, default=Path('/usr/lib'))
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.download:
        halogen_npu.download(args.root, args.models)
    files = halogen_npu.verify(args.root, args.models, full=True)
    required = sum(item['bytes'] for _, item, _ in halogen_npu.artifacts(args.root, args.models)) + 2 * 1024**3
    host = halogen_npu.host_errors(args.xrt_lib_dir, required_memlock=required)
    print(json.dumps({'release': halogen_npu.MANIFEST['release'], 'models': args.models, 'file_errors': files, 'host_errors': host, 'ready': not files and not host}, indent=2))
    # Downloads can be prepared successfully before the required host reboot.
    return int(bool(files))


if __name__ == '__main__':
    raise SystemExit(main())
