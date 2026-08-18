#!/usr/bin/env bash

set -Eeuo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
halo="$project_root/bin/halo-ai"
scope=all
profile_csv=""
config_path=""
output_root=/var/opt/halo-ai/state/benchmarks/office-profile-matrix
repetitions=3
prompt_tokens=4095,31998
completion_tokens=64
sample_id=66f37eb9821e116aacb2d295
smoke_only=false
run_container_tests=true

usage() {
    cat <<'EOF'
Usage: tests/office-profile-matrix.sh [options]

Run a long, sequential profile matrix with all inference inside managed Podman
containers. Successful elapsed times are persisted and used to order the next
run from fastest to slowest.

Options:
  --config FILE          Alternate halo-ai configuration file.
  --scope all|optimize   All ready profiles, or the active Qwen3.8/DS4 set.
  --profiles CSV         Exact comma-separated profile list; overrides --scope.
  --output-root DIR      Persistent results/history directory.
  --repetitions N        ROCmFPX exact-context repetitions. Default: 3.
  --prompt-tokens CSV    ROCmFPX prompt lengths. Default: 4095,31998.
  --completion-tokens N  ROCmFPX generated tokens. Default: 64.
  --sample-id ID         LongBench-v2 sample for non-ROCmFPX LLMs.
  --smoke-only           Start and smoke-test profiles without long benchmarks.
  --skip-container-tests Do not run the read-only Podman test suite first.
  -h, --help             Show this help.
EOF
}

while (($#)); do
    case "$1" in
        --config) config_path=${2:?--config requires a file}; shift 2 ;;
        --scope) scope=${2:?--scope requires all or optimize}; shift 2 ;;
        --profiles) profile_csv=${2:?--profiles requires CSV}; shift 2 ;;
        --output-root) output_root=${2:?--output-root requires a directory}; shift 2 ;;
        --repetitions) repetitions=${2:?--repetitions requires an integer}; shift 2 ;;
        --prompt-tokens) prompt_tokens=${2:?--prompt-tokens requires CSV}; shift 2 ;;
        --completion-tokens) completion_tokens=${2:?--completion-tokens requires an integer}; shift 2 ;;
        --sample-id) sample_id=${2:?--sample-id requires an ID}; shift 2 ;;
        --smoke-only) smoke_only=true; shift ;;
        --skip-container-tests) run_container_tests=false; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'error: unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$scope" == all || "$scope" == optimize ]] || {
    printf 'error: --scope must be all or optimize\n' >&2
    exit 2
}
[[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || {
    printf 'error: --repetitions must be positive\n' >&2
    exit 2
}
[[ "$completion_tokens" =~ ^[1-9][0-9]*$ ]] || {
    printf 'error: --completion-tokens must be positive\n' >&2
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
history="$output_root/profile-history.tsv"
mkdir -p -- "$run_dir"

snapshot_environment() {
    local destination=$1
    {
        printf 'recorded_at=%s\n' "$(date -Ins --utc)"
        printf 'boot_id=%s\n' "$(< /proc/sys/kernel/random/boot_id)"
        printf 'uptime=%s\n' "$(< /proc/uptime)"
        printf 'kernel=%s\n' "$(uname -r)"
        printf 'cmdline=%s\n' "$(< /proc/cmdline)"
        printf 'loadavg=%s\n' "$(< /proc/loadavg)"
        while IFS= read -r path; do
            printf '%s=%s\n' "$path" "$(< "$path")"
        done < <(find /sys/class/power_supply -maxdepth 2 -type f \
            \( -name online -o -name status -o -name type \) 2>/dev/null | sort)
        while IFS= read -r path; do
            printf '%s=%s\n' "$path" "$(< "$path")"
        done < <(find /sys/devices/system/cpu/cpufreq -maxdepth 2 -type f \
            \( -name scaling_governor -o -name energy_performance_preference \) \
            2>/dev/null | sort)
        if command -v sensors >/dev/null; then sensors 2>/dev/null || true; fi
    } >"$destination"
}

cleanup() {
    "${halo_cmd[@]}" stop all >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

snapshot_environment "$run_dir/environment-before.txt"
"${halo_cmd[@]}" status >"$run_dir/halo-status-before.json"
if "$run_container_tests"; then
    "$project_root/tests/container.sh" 2>&1 | tee "$run_dir/container-tests.log"
fi

declare -A ready engine seed_seconds last_seconds
while IFS=$'\t' read -r profile profile_engine availability _reason; do
    engine["$profile"]=$profile_engine
    if [[ "$availability" == ready ]]; then ready["$profile"]=1; fi
done < <("${halo_cmd[@]}" profiles list)

# Initial same-host wall-time ordering. Successful office runs supersede these
# seeds through profile-history.tsv.
seed_seconds[qwen3.6-35b-a3b-q8xl-lemonade]=18.7
seed_seconds[qwen3.6-35b-a3b-q8xl-128k-lemonade]=18.7
seed_seconds[qwen3.6-35b-a3b-q8xl-65k-lemonade]=18.9
seed_seconds[qwen3.6-35b-a3b-q8xl-128k-mtp-lemonade]=19.1
seed_seconds[qwen3.6-35b-a3b-q8xl-mtp-lemonade]=19.8
seed_seconds[qwen3.6-35b-a3b-q8xl-mtp-llamacpp]=20.1
seed_seconds[qwen3.6-27b-q8xl-128k-vision-lemonade]=48.5
seed_seconds[qwen3.6-27b-q8xl-vision-lemonade]=49.3
seed_seconds[qwen3.6-27b-q8xl-128k-lemonade]=49.1
seed_seconds[qwen3.6-27b-q8xl-lemonade]=50.1
seed_seconds[qwen3.6-27b-q8xl-mtp-lemonade]=52.4
seed_seconds[qwen3.6-27b-q8xl-vision-llamacpp]=52.5
seed_seconds[qwen3.6-27b-q8xl-mtp-llamacpp]=54.8
seed_seconds[ds4-deepseek-v4-flash-hybrid]=107.2
seed_seconds[ds4-deepseek-v4-flash-hybrid-kv]=107.7
seed_seconds[deepseek-v4-flash-0731-iq3xxs-llamacpp]=183.9
seed_seconds[deepseek-v4-flash-0731-iq3xxs-dspark-llamacpp]=283.7
seed_seconds[qwen38-27b-rocmfp4-baseline]=267
seed_seconds[qwen38-27b-rocmfp4-mtp-conservative-q5-draft]=296
seed_seconds[qwen38-27b-rocmfp4-mtp]=299
seed_seconds[qwen38-27b-rocmfp4-mtp-q5-draft]=304
seed_seconds[qwen38-27b-rocmfp8-baseline]=275
seed_seconds[qwen38-27b-rocmfp8-mtp]=305
seed_seconds[ds4-deepseek-v4-flash-hybrid-dspark-16k]=999
seed_seconds[seamless-m4t-v2-large-speech]=999

if [[ -r "$history" ]]; then
    while IFS=$'\t' read -r _recorded profile seconds result; do
        if [[ "$result" == pass && "$seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            last_seconds["$profile"]=$seconds
        fi
    done <"$history"
fi

requested=()
if [[ -n "$profile_csv" ]]; then
    IFS=, read -r -a requested <<<"$profile_csv"
elif [[ "$scope" == optimize ]]; then
    requested=(
        qwen38-27b-rocmfp4-baseline
        qwen38-27b-rocmfp4-mtp
        qwen38-27b-rocmfp4-mtp-q5-draft
        qwen38-27b-rocmfp4-mtp-conservative-q5-draft
        qwen38-27b-rocmfp8-baseline
        qwen38-27b-rocmfp8-mtp
        ds4-deepseek-v4-flash-hybrid
        ds4-deepseek-v4-flash-hybrid-kv
        ds4-deepseek-v4-flash-hybrid-dspark-16k
    )
else
    while IFS= read -r profile; do requested+=("$profile"); done < <(
        printf '%s\n' "${!engine[@]}" | sort
    )
fi

order_file="$run_dir/profile-order.tsv"
for profile in "${requested[@]}"; do
    if [[ -z "${engine[$profile]:-}" ]]; then
        printf 'unknown\t%s\n' "$profile" >>"$run_dir/skipped.tsv"
        continue
    fi
    if [[ -z "${ready[$profile]:-}" ]]; then
        printf 'disabled\t%s\n' "$profile" >>"$run_dir/skipped.tsv"
        continue
    fi
    score=${last_seconds[$profile]:-${seed_seconds[$profile]:-999999}}
    printf '%015.3f\t%s\n' "$score" "$profile" >>"$order_file"
done
sort -n -o "$order_file" "$order_file"

failures=0
while IFS=$'\t' read -r prior_seconds profile; do
    profile_dir="$run_dir/$profile"
    mkdir -p -- "$profile_dir"
    snapshot_environment "$profile_dir/environment-before.txt"
    started_epoch=$(date +%s)
    result=pass
    {
        printf 'profile=%s prior_order_seconds=%s engine=%s\n' \
            "$profile" "$prior_seconds" "${engine[$profile]}"
        "${halo_cmd[@]}" start "$profile" --switch
        "${halo_cmd[@]}" test "$profile"
        if ! "$smoke_only"; then
            if [[ "${engine[$profile]}" == rocmfpx ]]; then
                "${halo_cmd[@]}" bench rocmfpx-context "$profile" \
                    --prompt-pattern unique \
                    --prompt-tokens "$prompt_tokens" \
                    --completion-tokens "$completion_tokens" \
                    --repetitions "$repetitions" \
                    --output "$profile_dir/rocmfpx-context.json"
            elif [[ "${engine[$profile]}" != speech ]]; then
                "${halo_cmd[@]}" bench longbench-v2 run "$profile" \
                    --sample-id "$sample_id" --max-tokens 128 \
                    --output "$profile_dir/longbench-v2.jsonl"
            fi
        fi
    } > >(tee "$profile_dir/run.log") 2>&1 || result=fail
    "${halo_cmd[@]}" stop "$profile" >>"$profile_dir/run.log" 2>&1 || result=fail
    ended_epoch=$(date +%s)
    elapsed=$((ended_epoch - started_epoch))
    snapshot_environment "$profile_dir/environment-after.txt"
    printf '%s\t%s\t%s\t%s\n' "$(date -Ins --utc)" "$profile" "$elapsed" "$result" \
        >>"$history"
    printf '%s\t%s\t%s\n' "$profile" "$elapsed" "$result" >>"$run_dir/results.tsv"
    if [[ "$result" != pass ]]; then failures=$((failures + 1)); fi
done <"$order_file"

snapshot_environment "$run_dir/environment-after.txt"
"${halo_cmd[@]}" status >"$run_dir/halo-status-after.json"
printf 'run_dir=%s failures=%s\n' "$run_dir" "$failures"
((failures == 0))
