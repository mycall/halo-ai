#!/usr/bin/env bash

set -Eeuo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
test_tmp=$(mktemp -d)
trap 'rm -rf -- "$test_tmp"' EXIT

export PYTHONPYCACHEPREFIX="$test_tmp/pycache"

bash -n "$project_root/install.sh"
bash -n "$project_root/uninstall.sh"
bash -n "$project_root/bin/halo-ai"
grep -Fq 'cd -- "$run_home"' "$project_root/uninstall.sh" || {
    printf 'uninstaller does not enter the rootless operator home before Podman commands\n' >&2
    exit 1
}
if grep -Eq 'local .*link=.*temporary=.*\$\{link\}' "$project_root/install.sh"; then
    printf 'unsafe same-declaration local expansion found in install.sh\n' >&2
    exit 1
fi
for stage in directories release config links verify; do
    grep -Eq "^action_${stage}\\(\\)" "$project_root/install.sh" || {
        printf 'missing installer action for stage %s\n' "$stage" >&2
        exit 1
    }
    grep -Eq "^verify_${stage}\\(\\)" "$project_root/install.sh" || {
        printf 'missing installer verifier for stage %s\n' "$stage" >&2
        exit 1
    }
done
python3 -m py_compile "$project_root/lib/halo_ai/cli.py"
python3 -m json.tool "$project_root/config/models.d/strix-halo.json" >/dev/null
for preset in "$project_root"/config/request-presets.d/*.json; do
    python3 -m json.tool "$preset" >/dev/null
done
for result in "$project_root"/docs/results/*.json; do
    python3 -m json.tool "$result" >/dev/null
done

python3 -m unittest discover -s "$project_root/tests" -p 'test_*.py'
source_config="$project_root/config/halo-ai.env.example"
"$project_root/bin/halo-ai" --config "$source_config" profiles list >/dev/null
"$project_root/bin/halo-ai" --config "$source_config" presets list >/dev/null
"$project_root/bin/halo-ai" --config "$source_config" profiles render ds4-deepseek-v4-flash-hybrid | \
    grep -Fq -- '--mount type=bind,src=/srv/halo-ai/models/'
ln -s "$project_root/bin/halo-ai" "$test_tmp/halo-ai-symlink"
"$test_tmp/halo-ai-symlink" --config "$source_config" --version >/dev/null
"$project_root/install.sh" --run-user "$(id -un)" --dry-run >/dev/null
"$project_root/uninstall.sh" --run-user "$(id -un)" --dry-run --keep-podman >/dev/null

printf 'halo-ai smoke tests passed\n'
