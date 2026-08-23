#!/usr/bin/env bash

set -Eeuo pipefail

source_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
dry_run=false

usage() {
    cat <<'EOF'
Usage: ./reload.sh [--dry-run]

Redeploy this halo-ai source checkout with a fresh installer journal. The
script determines the rootless operator account and requests sudo itself.

Options:
  --dry-run   Preview the installer plan without sudo or host changes
  -h, --help  Show this help
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

if ((EUID == 0)); then
    run_user=${SUDO_USER:-}
    [[ -n "$run_user" && "$run_user" != root ]] || \
        die "run ./reload.sh as the non-root halo-ai operator"
else
    run_user=$(id -un)
fi
[[ "$run_user" =~ ^[a-zA-Z0-9_.-]+$ && "$run_user" != root ]] || \
    die "could not determine a valid non-root operator"

installer=(
    "$source_root/install.sh"
    --run-user "$run_user"
    --restart-journal
    --yes
)

if "$dry_run"; then
    exec "${installer[@]}" --dry-run
fi

if ((EUID == 0)); then
    exec "${installer[@]}"
fi

command -v sudo >/dev/null || die "sudo is required to reload halo-ai"
exec sudo "${installer[@]}"
