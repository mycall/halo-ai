#!/usr/bin/env bash

set -Eeuo pipefail

if [[ "${HALO_AI_QUALITY_SLEEP_INHIBITED:-0}" != 1 ]] && command -v systemd-inhibit >/dev/null; then
    exec systemd-inhibit \
        --what=sleep --mode=block --who=halo-ai \
        --why="ROCmFPX fixed quality and identity matrix" \
        env HALO_AI_QUALITY_SLEEP_INHIBITED=1 bash "$0" "$@"
fi

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
halo="$project_root/bin/halo-ai"
config_path=""
output_root=/var/opt/halo-ai/state/benchmarks/rocmfpx-quality
process_repetitions=2
profile_csv="qwen38-27b-rocmfp4-baseline,qwen38-27b-rocmfp8-baseline"
run_container_tests=true

usage() {
    cat <<'EOF'
Usage: tests/rocmfpx-quality-matrix.sh [options]

Run the fixed, download-free ROCmFPX quality suite in fresh managed Podman
processes and compare cross-process output-token identity.

Options:
  --config FILE              Alternate halo-ai configuration file.
  --output-root DIR          Persistent result directory.
  --process-repetitions N    Fresh starts per profile. Default: 2.
  --profiles CSV             Exact ROCmFPX profiles to run.
  --skip-container-tests     Skip the source suite preflight.
  -h, --help                 Show this help.
EOF
}

while (($#)); do
    case "$1" in
        --config) config_path=${2:?--config requires a file}; shift 2 ;;
        --output-root) output_root=${2:?--output-root requires a directory}; shift 2 ;;
        --process-repetitions) process_repetitions=${2:?--process-repetitions requires an integer}; shift 2 ;;
        --profiles) profile_csv=${2:?--profiles requires CSV}; shift 2 ;;
        --skip-container-tests) run_container_tests=false; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'error: unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$process_repetitions" =~ ^[1-9][0-9]*$ ]] && ((process_repetitions <= 10)) || {
    printf 'error: --process-repetitions must be in [1, 10]\n' >&2
    exit 2
}
[[ "$output_root" == /* && "$output_root" != / ]] || {
    printf 'error: --output-root must be a safe absolute path\n' >&2
    exit 2
}

config_args=()
if [[ -n "$config_path" ]]; then
    config_path=$(readlink -m -- "$config_path")
    config_args=(--config "$config_path")
fi
halo_cmd=("$halo" "${config_args[@]}")
run_id=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$output_root/$run_id"
mkdir -p -- "$run_dir"

cleanup() {
    "${halo_cmd[@]}" stop all >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

if command -v powerprofilesctl >/dev/null; then
    powerprofilesctl set performance
fi
if "$run_container_tests"; then
    "$project_root/tests/container.sh" 2>&1 | tee "$run_dir/container-tests.log"
fi

IFS=, read -r -a profiles <<<"$profile_csv"
records=()
failures=0
for repetition in $(seq 1 "$process_repetitions"); do
    for profile in "${profiles[@]}"; do
        profile_dir="$run_dir/$profile/process-$repetition"
        mkdir -p -- "$profile_dir"
        result=pass
        {
            printf 'profile=%s process_repetition=%s\n' "$profile" "$repetition" &&
            "${halo_cmd[@]}" start "$profile" --switch &&
            "${halo_cmd[@]}" test "$profile" &&
            "${halo_cmd[@]}" bench rocmfpx-quality "$profile" \
                --process-repetition "$repetition" \
                --output "$profile_dir/quality.json"
        } > >(tee "$profile_dir/run.log") 2>&1 || result=fail
        "${halo_cmd[@]}" stop "$profile" >>"$profile_dir/run.log" 2>&1 || result=fail
        printf '%s\t%s\t%s\n' "$profile" "$repetition" "$result" >>"$run_dir/results.tsv"
        if [[ "$result" == pass ]]; then
            records+=("$profile_dir/quality.json")
        else
            failures=$((failures + 1))
        fi
    done
done

if ((failures == 0)); then
    "${halo_cmd[@]}" tune quality-compare "${records[@]}" \
        --output "$run_dir/comparison.json" | tee "$run_dir/comparison.log"
else
    printf 'error: refusing comparison because %s process run(s) failed\n' "$failures" >&2
fi

printf 'run_dir=%s failures=%s\n' "$run_dir" "$failures"
((failures == 0))
