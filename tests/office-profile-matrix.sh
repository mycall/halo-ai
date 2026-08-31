#!/usr/bin/env bash

set -Eeuo pipefail

# Suspend invalidates wall-time, throughput, and power observations. Hold a
# transient host inhibitor for the complete run; it disappears with this
# process and installs no service or package.
if [[ "${HALO_AI_OFFICE_SLEEP_INHIBITED:-0}" != 1 ]] && command -v systemd-inhibit >/dev/null; then
    exec systemd-inhibit \
        --what=sleep --mode=block --who=halo-ai \
        --why="controlled office profile matrix" \
        env HALO_AI_OFFICE_SLEEP_INHIBITED=1 bash "$0" "$@"
fi

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
halo="$project_root/bin/halo-ai"
scope=all
profile_csv=""
exclude_profile_csv=""
config_path=""
output_root=/var/opt/halo-ai/state/benchmarks/office-profile-matrix
repetitions=3
prompt_tokens=4095,31998
completion_tokens=64
sample_id=66f37eb9821e116aacb2d295
smoke_only=false
run_container_tests=true
required_power_profile=performance
minimum_observed_ppt_watts=45
power_sampler_pid=""
run_peak_microwatts=0
last_peak_microwatts=0

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
  --exclude-profiles CSV Skip exact comma-separated profiles and record them.
  --output-root DIR      Persistent results/history directory.
  --repetitions N        ROCmFPX exact-context repetitions. Default: 3.
  --prompt-tokens CSV    ROCmFPX prompt lengths. Default: 4095,31998.
  --completion-tokens N  ROCmFPX generated tokens. Default: 64.
  --sample-id ID         LongBench-v2 sample for non-ROCmFPX LLMs.
  --minimum-ppt-watts N  Require the run to observe APU PPT above N watts.
                         Default: 45.
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
        --exclude-profiles) exclude_profile_csv=${2:?--exclude-profiles requires CSV}; shift 2 ;;
        --output-root) output_root=${2:?--output-root requires a directory}; shift 2 ;;
        --repetitions) repetitions=${2:?--repetitions requires an integer}; shift 2 ;;
        --prompt-tokens) prompt_tokens=${2:?--prompt-tokens requires CSV}; shift 2 ;;
        --completion-tokens) completion_tokens=${2:?--completion-tokens requires an integer}; shift 2 ;;
        --sample-id) sample_id=${2:?--sample-id requires an ID}; shift 2 ;;
        --minimum-ppt-watts) minimum_observed_ppt_watts=${2:?--minimum-ppt-watts requires an integer}; shift 2 ;;
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
[[ "$minimum_observed_ppt_watts" =~ ^[1-9][0-9]*$ ]] || {
    printf 'error: --minimum-ppt-watts must be positive\n' >&2
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

power_profile() {
    powerprofilesctl get 2>/dev/null || true
}

platform_profile() {
    if [[ -r /sys/firmware/acpi/platform_profile ]]; then
        cat /sys/firmware/acpi/platform_profile
    fi
}

suspend_offset_milliseconds() {
    python3 - <<'PY'
import time

print(round((time.clock_gettime(time.CLOCK_BOOTTIME) - time.monotonic()) * 1000))
PY
}

assert_performance_power() {
    local reason=$1
    local pp platform
    pp=$(power_profile)
    platform=$(platform_profile)
    if [[ "$pp" != "$required_power_profile" || "$platform" != "$required_power_profile" ]]; then
        printf 'error: %s: expected power and platform profiles to be performance; got power=%s platform=%s\n' \
            "$reason" "${pp:-unavailable}" "${platform:-unavailable}" >&2
        return 1
    fi
}

ppt_sensor=""
for candidate in /sys/class/hwmon/hwmon*/power1_average; do
    [[ -r "$candidate" ]] || continue
    if [[ "$(< "${candidate%/*}/name")" == amdgpu ]]; then
        ppt_sensor=$candidate
        break
    fi
done

start_power_sampler() {
    local destination=$1
    [[ -n "$ppt_sensor" ]] || return 0
    (
        while :; do
            printf '%s\t%s\n' "$(date -Ins --utc)" "$(< "$ppt_sensor")"
            sleep 1
        done
    ) >"$destination" &
    power_sampler_pid=$!
}

stop_power_sampler() {
    local source=$1
    local peak=0
    if [[ -n "$power_sampler_pid" ]]; then
        kill "$power_sampler_pid" 2>/dev/null || true
        wait "$power_sampler_pid" 2>/dev/null || true
        power_sampler_pid=""
    fi
    if [[ -s "$source" ]]; then
        peak=$(awk 'BEGIN { max=0 } $2 + 0 > max { max=$2 + 0 } END { printf "%.0f", max }' "$source")
    fi
    last_peak_microwatts=$peak
    if ((peak > run_peak_microwatts)); then run_peak_microwatts=$peak; fi
}

validate_ds4_dspark_stats() {
    local source=$1
    local line proposed accepted
    line=$(rg 'DSpark stats cycles=' "$source" | tail -1 || true)
    if [[ -z "$line" ]]; then
        printf 'error: DS4 did not flush DSpark runtime statistics\n' >&2
        return 1
    fi
    if [[ "$line" =~ proposed=([0-9]+) ]]; then proposed=${BASH_REMATCH[1]}; else proposed=0; fi
    if [[ "$line" =~ accepted_draft=([0-9]+) ]]; then accepted=${BASH_REMATCH[1]}; else accepted=0; fi
    printf '%s\n' "$line"
    if ((proposed == 0 || accepted == 0)); then
        printf 'error: DS4 DSpark produced no accepted proposals (proposed=%s accepted=%s)\n' \
            "$proposed" "$accepted" >&2
        return 1
    fi
}

snapshot_environment() {
    local destination=$1
    {
        printf 'recorded_at=%s\n' "$(date -Ins --utc)"
        printf 'boot_id=%s\n' "$(< /proc/sys/kernel/random/boot_id)"
        printf 'uptime=%s\n' "$(< /proc/uptime)"
        printf 'kernel=%s\n' "$(uname -r)"
        printf 'cmdline=%s\n' "$(< /proc/cmdline)"
        printf 'loadavg=%s\n' "$(< /proc/loadavg)"
        printf 'suspend_offset_milliseconds=%s\n' "$(suspend_offset_milliseconds)"
        printf 'power_profile=%s\n' "$(power_profile)"
        printf 'platform_profile=%s\n' "$(platform_profile)"
        if [[ -n "$ppt_sensor" ]]; then
            printf 'apu_ppt_microwatts=%s\n' "$(< "$ppt_sensor")"
        fi
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
    if [[ -n "$power_sampler_pid" ]]; then
        kill "$power_sampler_pid" 2>/dev/null || true
        wait "$power_sampler_pid" 2>/dev/null || true
    fi
    "${halo_cmd[@]}" stop all >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# This is the strongest state exposed by the installed amd-pstate/ACPI drivers.
# Numeric GPD WIN 5 DPTC controls are not available on this kernel, so a PPT
# sampler below independently proves whether the loaded run crosses the target.
command -v powerprofilesctl >/dev/null || {
    printf 'error: powerprofilesctl is required for an office matrix\n' >&2
    exit 1
}
powerprofilesctl set "$required_power_profile"
assert_performance_power preflight
snapshot_environment "$run_dir/environment-before.txt"
"${halo_cmd[@]}" status >"$run_dir/halo-status-before.json"
if "$run_container_tests"; then
    "$project_root/tests/container.sh" 2>&1 | tee "$run_dir/container-tests.log"
fi
assert_performance_power post-container-tests

declare -A ready engine seed_seconds last_seconds
while IFS=$'\t' read -r profile profile_engine availability _reason; do
    [[ "$_reason" == alias\ for\ * ]] && continue
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
seed_seconds[qwen3.8-27b-rocmfp4-baseline]=267
seed_seconds[qwen3.8-27b-rocmfp4-mtp-conservative-q5-draft]=296
seed_seconds[qwen3.8-27b-rocmfp4-mtp]=299
seed_seconds[qwen3.8-27b-rocmfp4-mtp-q5-draft]=304
seed_seconds[qwen3.8-27b-rocmfp8-baseline]=275
seed_seconds[qwen3.8-27b-rocmfp8-mtp]=305
seed_seconds[ds4-deepseek-v4-flash-hybrid-dspark-16k]=999
seed_seconds[seamless-m4t-v2-large-speech]=999

if [[ -r "$history" ]]; then
    while IFS=$'\t' read -r _recorded profile seconds result; do
        if [[ "$result" == pass && "$seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            last_seconds["$profile"]=$seconds
        elif [[ "$result" == invalid-* ]]; then
            unset "last_seconds[$profile]"
        fi
    done <"$history"
fi

requested=()
declare -A excluded
if [[ -n "$exclude_profile_csv" ]]; then
    IFS=, read -r -a excluded_profiles <<<"$exclude_profile_csv"
    for profile in "${excluded_profiles[@]}"; do excluded["$profile"]=1; done
fi
if [[ -n "$profile_csv" ]]; then
    IFS=, read -r -a requested <<<"$profile_csv"
elif [[ "$scope" == optimize ]]; then
    requested=(
        qwen3.8-27b-rocmfp4-baseline
        qwen3.8-27b-rocmfp4-mtp
        qwen3.8-27b-rocmfp4-mtp-q5-draft
        qwen3.8-27b-rocmfp4-mtp-conservative-q5-draft
        qwen3.8-27b-rocmfp8-baseline
        qwen3.8-27b-rocmfp8-mtp
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
    if [[ -n "${excluded[$profile]:-}" ]]; then
        printf 'excluded\t%s\n' "$profile" >>"$run_dir/skipped.tsv"
        continue
    fi
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
    assert_performance_power "$profile preflight"
    suspend_offset_before=$(suspend_offset_milliseconds)
    started_epoch=$(date +%s)
    result=pass
    power_samples="$profile_dir/apu-ppt.tsv"
    start_power_sampler "$power_samples"
    {
        printf 'profile=%s prior_order_seconds=%s engine=%s\n' \
            "$profile" "$prior_seconds" "${engine[$profile]}" &&
        "${halo_cmd[@]}" start "$profile" --switch &&
        "${halo_cmd[@]}" test "$profile" &&
        if "$smoke_only" || [[ "${engine[$profile]}" == speech ]]; then
            true
        elif [[ "${engine[$profile]}" == llamacpp ||
                "${engine[$profile]}" == rocmfpx ||
                "${engine[$profile]}" == lemonade ||
                "${engine[$profile]}" == strixvulkan ]]; then
            "${halo_cmd[@]}" bench rocmfpx-context "$profile" \
                --prompt-pattern unique \
                --prompt-tokens "$prompt_tokens" \
                --completion-tokens "$completion_tokens" \
                --repetitions "$repetitions" \
                --output "$profile_dir/rocmfpx-context.json"
        else
            "${halo_cmd[@]}" bench longbench-v2 run "$profile" \
                --sample-id "$sample_id" --max-tokens 128 \
                --output "$profile_dir/longbench-v2.jsonl"
        fi
    } > >(tee "$profile_dir/run.log") 2>&1 || result=fail
    "${halo_cmd[@]}" stop "$profile" >>"$profile_dir/run.log" 2>&1 || result=fail
    if [[ "$profile" == ds4-deepseek-v4-flash-hybrid-dspark-* ]]; then
        podman logs halo-ds4 >"$profile_dir/ds4-container.log" 2>&1 || result=fail
        validate_ds4_dspark_stats "$profile_dir/ds4-container.log" 2>&1 \
            | tee -a "$profile_dir/run.log" || result=fail
    fi
    stop_power_sampler "$power_samples"
    peak_microwatts=$last_peak_microwatts
    peak_watts=$(awk -v value="$peak_microwatts" 'BEGIN { printf "%.3f", value / 1000000 }')
    printf 'peak_apu_ppt_watts=%s\n' "$peak_watts" | tee -a "$profile_dir/run.log"
    assert_performance_power "$profile postflight" || result=fail
    suspend_offset_after=$(suspend_offset_milliseconds)
    suspend_delta_ms=$((suspend_offset_after - suspend_offset_before))
    if ((suspend_delta_ms > 1000)); then
        result=invalid-suspend
        printf 'invalid: suspend offset increased by %s ms during profile\n' \
            "$suspend_delta_ms" | tee -a "$profile_dir/run.log" >&2
        printf '%s\t%s\t%s\n' "$profile" "$suspend_delta_ms" "suspend detected" \
            >>"$run_dir/invalidations.tsv"
    fi
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
run_peak_watts=$(awk -v value="$run_peak_microwatts" 'BEGIN { printf "%.3f", value / 1000000 }')
printf 'peak_apu_ppt_watts=%s\n' "$run_peak_watts" >"$run_dir/power-summary.txt"
minimum_microwatts=$((minimum_observed_ppt_watts * 1000000))
if ! "$smoke_only" && ((run_peak_microwatts <= minimum_microwatts)); then
    printf 'error: run never observed APU PPT above %s W (peak %s W)\n' \
        "$minimum_observed_ppt_watts" "$run_peak_watts" | tee -a "$run_dir/power-summary.txt" >&2
    failures=$((failures + 1))
fi
printf 'run_dir=%s failures=%s peak_apu_ppt_watts=%s\n' "$run_dir" "$failures" "$run_peak_watts"
((failures == 0))
