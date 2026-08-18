#!/usr/bin/env bash

set -Eeuo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
test_image='docker.io/kyuz0/amd-strix-halo-toolboxes@sha256:8cdde6d42ab621b3d0f1e02618f4234c58b7036bec984a2a64fa205d38f99922'
run_uid=$(id -u)
run_gid=$(id -g)
run_user=$(id -un)
run_group=$(id -gn)

exec podman run --rm \
    --network none \
    --read-only \
    --userns keep-id \
    --user "${run_uid}:${run_gid}" \
    --passwd-entry "${run_user}:x:${run_uid}:${run_gid}:Halo test operator:/tmp:/sbin/nologin" \
    --group-entry "${run_group}:x:${run_gid}:" \
    --tmpfs /tmp:rw,noexec,nosuid,nodev,size=2g \
    --mount "type=bind,src=${project_root},dst=/workspace,ro" \
    --workdir /workspace \
    --entrypoint /usr/bin/bash \
    "$test_image" \
    ./tests/smoke.sh
