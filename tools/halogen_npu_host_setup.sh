#!/usr/bin/env bash
# Prepare CachyOS for optional Halogen NPU profiles. Does not reboot.
set -Eeuo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
upstream_revision=7f31bbd4021f217a1be9776bdb7304bcf8eca62d
if [[ ${1:-} == --dry-run ]]; then
    printf '%s\n' \
        'Install xrt and xrt-plugin-amdxdna using existing pacman repositories.' \
        'Configure unlimited memlock for the invoking non-root operator; existing sessions may need renewal.' \
        'Fetch and checksum the two upstream 0.16.2 fabric-clock files.' \
        'Install and enable the fabric-clock service for the next boot (not started now).' \
        'Stage the npu IOMMU host profile using the snapshot/backup lifecycle.' \
        'Preserve GTT/TTM and all unrelated boot arguments. No automatic reboot.'
    exec "$project_root/bin/halo-ai" host-profile set npu --dry-run
fi
[[ $# == 0 && $EUID == 0 ]] || { echo 'Usage: pkexec /usr/bin/bash tools/halogen_npu_host_setup.sh (or --dry-run without root)' >&2; exit 1; }
operator_uid=${PKEXEC_UID:-${SUDO_UID:-}}
[[ -n $operator_uid && $operator_uid != 0 ]] || { echo 'Invoke through pkexec or sudo from the non-root operator account.' >&2; exit 1; }
operator_name=$(id -un "$operator_uid")
python3 "$project_root/tools/halogen_memlock_setup.py" --user "$operator_name"
temporary=$(mktemp -d)
trap 'rm -rf -- "$temporary"' EXIT
for name in halogen-fabric-clock halogen-fabric-clock.service; do
    curl --fail --silent --show-error --location \
        "https://raw.githubusercontent.com/peonist-ai/halogen-flash-server/$upstream_revision/deploy/host/$name" \
        --output "$temporary/$name"
done
(
    cd "$temporary"
    sha256sum --check <<'CHECKSUMS'
a3a4f7f4e0142e71ac1390415f9bf418064f6b3b962dc43112e071306a0ac281  halogen-fabric-clock
5f9f1cc4c85a1c027e907aa355bbabc87d40eba544512b51146b543a49bf8a71  halogen-fabric-clock.service
CHECKSUMS
)
pacman -S --needed --noconfirm xrt xrt-plugin-amdxdna
install -m 0755 "$temporary/halogen-fabric-clock" /usr/local/sbin/halogen-fabric-clock
install -m 0644 "$temporary/halogen-fabric-clock.service" /etc/systemd/system/halogen-fabric-clock.service
systemctl daemon-reload
systemctl enable halogen-fabric-clock.service
"$project_root/bin/halo-ai" host-profile set npu --yes
printf '%s\n' 'Prepared. Reboot manually with firmware IOMMU enabled, then run halo-ai host-profile status and halogen-fabric-clock status.'
