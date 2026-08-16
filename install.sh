#!/usr/bin/env bash

set -Eeuo pipefail

readonly INSTALL_ROOT="/opt/halo-ai"
readonly RELEASES_ROOT="$INSTALL_ROOT/releases"
readonly CURRENT_LINK="$INSTALL_ROOT/current"
readonly COMMAND_LINK="/usr/local/bin/halo-ai"
readonly CONFIG_ROOT="/etc/opt/halo-ai"
readonly STATE_ROOT="/var/opt/halo-ai"
readonly CACHE_ROOT="/var/cache/halo-ai"
readonly JOURNAL_ROOT="/var/lib/halo-ai-installer"
readonly JOURNAL="$JOURNAL_ROOT/install-state.env"
readonly SNAPSHOT_CONFIG="root"

source_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
run_user=""
assume_yes=false
dry_run=false
use_snapper=true
restart_journal=false
repair_stage=""
active_stage="preflight"
pre_snapshot=""
post_snapshot=""
completed_stages=""

usage() {
    cat <<'EOF'
Usage: sudo ./install.sh --run-user USER [options]

Install or resume installation of halo-ai under /opt/halo-ai.

Options:
  --run-user USER       Rootless Podman operator; root is forbidden
  --dry-run             Print the resolved stages without changing the host
  --yes                 Skip the exact interactive confirmation
  --no-snapshot         Explicitly install without a Snapper pre/post pair
  --restart-journal     Archive an existing journal and revalidate every stage
  --repair-stage NAME   Reapply one completed stage after reporting drift
  -h, --help            Show this help

Rerunning the same command resumes automatically from the root-only journal at
/var/lib/halo-ai-installer/install-state.env. Completed stages are verified before they
are skipped. A failed stage is never marked complete.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

warn() {
    printf 'warning: %s\n' "$*" >&2
}

while (($#)); do
    case "$1" in
        --run-user)
            (($# >= 2)) || die "--run-user requires an argument"
            run_user=$2
            shift 2
            ;;
        --dry-run) dry_run=true; shift ;;
        --yes) assume_yes=true; shift ;;
        --no-snapshot) use_snapper=false; shift ;;
        --restart-journal) restart_journal=true; shift ;;
        --repair-stage)
            (($# >= 2)) || die "--repair-stage requires an argument"
            repair_stage=$2
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

if [[ -z "$run_user" && -n "${SUDO_USER:-}" && "$SUDO_USER" != root ]]; then
    run_user=$SUDO_USER
fi
[[ -n "$run_user" ]] || die "pass --run-user USER"
[[ "$run_user" =~ ^[a-zA-Z0-9_.-]+$ ]] || die "invalid run user: $run_user"
[[ "$run_user" != root ]] || die "HALO_AI_RUN_USER must not be root"
passwd_entry=$(getent passwd "$run_user") || die "unknown run user: $run_user"
IFS=: read -r _ _ run_uid run_gid _ run_home _ <<<"$passwd_entry"
[[ "$run_uid" =~ ^[0-9]+$ && "$run_gid" =~ ^[0-9]+$ ]] || die "invalid account entry for $run_user"

if ! "$dry_run" && ((EUID != 0)); then
    die "run as root (normally: sudo ./install.sh --run-user $run_user)"
fi

for required in python3 install cp mv ln readlink findmnt; do
    command -v "$required" >/dev/null || die "required command is missing: $required"
done

for required_path in bin/halo-ai lib/halo_ai/cli.py lib/halo_ai/longbench.py config/halo-ai.env.example \
    config/models.d/strix-halo.json docs/halo-ai.md uninstall.sh; do
    [[ -e "$source_root/$required_path" ]] || die "source checkout is incomplete: $required_path"
done

release_hash=$(python3 - "$source_root" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
for relative in ("bin", "lib", "config", "docs"):
    for path in sorted((root / relative).rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
for relative in ("install.sh", "uninstall.sh"):
    path = root / relative
    digest.update(relative.encode() + b"\0" + path.read_bytes())
print(digest.hexdigest()[:16])
PY
)
release_id="0.1.0-$release_hash"
release_root="$RELEASES_ROOT/$release_id"
requested_release_id=$release_id
resuming_pinned_release=false
archive_journal=false
journal_archive=""

normalize() { readlink -m -- "$1"; }
for fixed in "$INSTALL_ROOT" "$RELEASES_ROOT" "$CONFIG_ROOT" "$STATE_ROOT" "$CACHE_ROOT" "$JOURNAL_ROOT"; do
    normalized=$(normalize "$fixed")
    [[ "$normalized" == "$fixed" && "$normalized" != / ]] || die "unsafe fixed target: $fixed -> $normalized"
done

configured_model_root="/srv/halo-ai/models"
if [[ -r "$CONFIG_ROOT/config.env" ]]; then
    configured_model_root=$(sed -n \
        's/^[[:space:]]*HALO_AI_MODELS_ROOT[[:space:]]*=[[:space:]]*//p' \
        "$CONFIG_ROOT/config.env" | tail -n 1)
    configured_model_root=${configured_model_root%\"}; configured_model_root=${configured_model_root#\"}
    configured_model_root=${configured_model_root%\'}; configured_model_root=${configured_model_root#\'}
fi
[[ "$configured_model_root" == /* && "$configured_model_root" != / ]] || die "unsafe configured model root"
for target in "$INSTALL_ROOT" "$CONFIG_ROOT" "$STATE_ROOT" "$CACHE_ROOT"; do
    target_normalized=$(normalize "$target")
    model_normalized=$(normalize "$configured_model_root")
    if [[ "$model_normalized" == "$target_normalized" ||
          "$model_normalized" == "$target_normalized/"* ||
          "$target_normalized" == "$model_normalized/"* ]]; then
        die "install target overlaps external model root: $target and $configured_model_root"
    fi
done

read_journal_value() {
    local key=$1
    [[ -r "$JOURNAL" ]] || return 1
    sed -n "s/^${key}=//p" "$JOURNAL" | tail -n 1
}

write_journal() {
    local status=$1
    local failed=${2:-}
    local temporary="$JOURNAL_ROOT/.install-state.$$.tmp"

    if "$dry_run"; then return 0; fi
    umask 077
    {
        printf 'JOURNAL_VERSION=1\n'
        printf 'RELEASE_ID=%s\n' "$release_id"
        printf 'SOURCE_HASH=%s\n' "$release_hash"
        printf 'RUN_USER=%s\n' "$run_user"
        printf 'RUN_UID=%s\n' "$run_uid"
        printf 'STATUS=%s\n' "$status"
        printf 'COMPLETED_STAGES=%s\n' "$completed_stages"
        printf 'FAILED_STAGE=%s\n' "$failed"
        printf 'UPDATED_AT=%(%Y-%m-%dT%H:%M:%SZ)T\n' -1
    } >"$temporary"
    chmod 0600 "$temporary"
    mv -fT -- "$temporary" "$JOURNAL"
    sync -f "$JOURNAL"
}

stage_complete() {
    local stage=$1
    [[ ",$completed_stages," == *",$stage,"* ]]
}

mark_complete() {
    local stage=$1
    if ! stage_complete "$stage"; then
        completed_stages=${completed_stages:+$completed_stages,}$stage
    fi
    write_journal running
}

run_action() {
    if "$dry_run"; then
        printf '  '
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

verify_directories() {
    [[ -d "$RELEASES_ROOT" && -d "$CONFIG_ROOT/models.d" && -d "$JOURNAL_ROOT" &&
       -d "$CONFIG_ROOT/request-presets.d" && -d "$STATE_ROOT/state" &&
       -d "$CACHE_ROOT" && -d "$CACHE_ROOT/ds4-kv" && -d "$CACHE_ROOT/speech" ]]
}

action_directories() {
    run_action install -d -m 0755 -o root -g root \
        "$INSTALL_ROOT" "$RELEASES_ROOT" "$CONFIG_ROOT" \
        "$CONFIG_ROOT/models.d" "$CONFIG_ROOT/request-presets.d"
    run_action install -d -m 0700 -o root -g root "$JOURNAL_ROOT"
    run_action install -d -m 0750 -o "$run_uid" -g "$run_gid" \
        "$STATE_ROOT" "$STATE_ROOT/state" "$CACHE_ROOT" "$CACHE_ROOT/ds4-kv" \
        "$CACHE_ROOT/speech"
}

verify_release() {
    [[ -x "$release_root/bin/halo-ai" && -r "$release_root/lib/halo_ai/cli.py" &&
       -r "$release_root/lib/halo_ai/longbench.py" &&
       -r "$release_root/docs/halo-ai.md" && -x "$release_root/uninstall.sh" &&
       -r "$release_root/.source-hash" && "$(<"$release_root/.source-hash")" == "$release_hash" ]]
}

action_release() {
    local partial="$RELEASES_ROOT/.$release_id.partial"
    if verify_release; then
        return 0
    fi
    [[ ! -e "$release_root" ]] || die \
        "release directory exists but fails its source-hash verification: $release_root"
    if [[ -e "$partial" ]]; then
        [[ "$partial" == "$RELEASES_ROOT/."* ]] || die "unsafe partial release path"
        run_action rm -rf --one-file-system -- "$partial"
    fi
    run_action install -d -m 0755 -o root -g root "$partial"
    if ! "$dry_run"; then
        cp -a -- "$source_root/bin" "$source_root/lib" "$source_root/config" \
            "$source_root/docs" "$source_root/install.sh" "$source_root/uninstall.sh" "$partial/"
        find "$partial" -type d -name __pycache__ -prune -exec rm -rf -- {} +
        find "$partial" -type f -name '*.pyc' -delete
        printf '%s\n' "$release_hash" >"$partial/.source-hash"
        chown -R root:root "$partial"
        find "$partial" -type d -exec chmod 0755 {} +
        chmod 0755 "$partial/bin/halo-ai" "$partial/install.sh" "$partial/uninstall.sh"
        mv -T -- "$partial" "$release_root"
    fi
}

verify_config() {
    [[ -r "$CONFIG_ROOT/config.env" && -r "$CONFIG_ROOT/models.d/00-strix-halo.json" &&
       -r "$CONFIG_ROOT/request-presets.d/qwen3.6-planner-coding.json" &&
       -r "$CONFIG_ROOT/request-presets.d/qwen3.6-implementer.json" ]] || return 1
    grep -Eq "^[[:space:]]*HALO_AI_RUN_USER[[:space:]]*=[[:space:]]*${run_user}[[:space:]]*$" \
        "$CONFIG_ROOT/config.env" &&
        grep -Eq '^[[:space:]]*LEMONADE_LLAMACPP_ROCM_BIN[[:space:]]*=[[:space:]]*(builtin|latest|b[0-9]+)[[:space:]]*$' \
            "$CONFIG_ROOT/config.env" &&
        grep -Eq '^[[:space:]]*LLAMACPP_IMAGE[[:space:]]*=[[:space:]]*[^[:space:]]+[[:space:]]*$' \
            "$CONFIG_ROOT/config.env" &&
        grep -Eq '^[[:space:]]*SPEECH_IMAGE[[:space:]]*=[[:space:]]*[^[:space:]]+[[:space:]]*$' \
            "$CONFIG_ROOT/config.env" &&
        runuser -u "$run_user" -- test -r "$CONFIG_ROOT/config.env"
}

action_config() {
    if [[ ! -e "$CONFIG_ROOT/config.env" ]]; then
        if "$dry_run"; then
            printf '  create %s with HALO_AI_RUN_USER=%s\n' "$CONFIG_ROOT/config.env" "$run_user"
        else
            sed "s/^HALO_AI_RUN_USER=.*/HALO_AI_RUN_USER=$run_user/" \
                "$source_root/config/halo-ai.env.example" >"$CONFIG_ROOT/.config.env.tmp"
            chmod 0640 "$CONFIG_ROOT/.config.env.tmp"
            chown root:"$run_gid" "$CONFIG_ROOT/.config.env.tmp"
            mv -T "$CONFIG_ROOT/.config.env.tmp" "$CONFIG_ROOT/config.env"
        fi
    elif ! grep -Eq '^[[:space:]]*LEMONADE_LLAMACPP_ROCM_BIN[[:space:]]*=' "$CONFIG_ROOT/config.env"; then
        if "$dry_run"; then
            printf '  append LEMONADE_LLAMACPP_ROCM_BIN=latest to %s\n' "$CONFIG_ROOT/config.env"
        else
            cp -- "$CONFIG_ROOT/config.env" "$CONFIG_ROOT/.config.env.tmp"
            printf '\n# Resolve the newest stable-channel ROCm package on each explicit start.\n' \
                >>"$CONFIG_ROOT/.config.env.tmp"
            printf 'LEMONADE_LLAMACPP_ROCM_BIN=latest\n' >>"$CONFIG_ROOT/.config.env.tmp"
            chmod 0640 "$CONFIG_ROOT/.config.env.tmp"
            chown root:"$run_gid" "$CONFIG_ROOT/.config.env.tmp"
            mv -T "$CONFIG_ROOT/.config.env.tmp" "$CONFIG_ROOT/config.env"
        fi
    fi
    if ! grep -Eq '^[[:space:]]*LLAMACPP_IMAGE[[:space:]]*=[[:space:]]*[^[:space:]]+[[:space:]]*$' \
        "$CONFIG_ROOT/config.env"; then
        if "$dry_run"; then
            printf '  set LLAMACPP_IMAGE=docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.14 in %s\n' \
                "$CONFIG_ROOT/config.env"
        else
            cp -- "$CONFIG_ROOT/config.env" "$CONFIG_ROOT/.config.env.tmp"
            if grep -Eq '^[[:space:]]*LLAMACPP_IMAGE[[:space:]]*=' "$CONFIG_ROOT/.config.env.tmp"; then
                sed -i \
                    's|^[[:space:]]*LLAMACPP_IMAGE[[:space:]]*=.*$|LLAMACPP_IMAGE=docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.14|' \
                    "$CONFIG_ROOT/.config.env.tmp"
            else
                printf '\nLLAMACPP_IMAGE=docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.14\n' \
                    >>"$CONFIG_ROOT/.config.env.tmp"
            fi
            chmod 0640 "$CONFIG_ROOT/.config.env.tmp"
            chown root:"$run_gid" "$CONFIG_ROOT/.config.env.tmp"
            mv -T "$CONFIG_ROOT/.config.env.tmp" "$CONFIG_ROOT/config.env"
        fi
    fi
    if ! grep -Eq '^[[:space:]]*SPEECH_IMAGE[[:space:]]*=[[:space:]]*[^[:space:]]+[[:space:]]*$' \
        "$CONFIG_ROOT/config.env"; then
        if "$dry_run"; then
            printf '  append Seamless speech runtime settings to %s\n' "$CONFIG_ROOT/config.env"
        else
            cp -- "$CONFIG_ROOT/config.env" "$CONFIG_ROOT/.config.env.tmp"
            printf '\n# Project-built ROCm 7.14 Seamless M4T v2 service.\n' \
                >>"$CONFIG_ROOT/.config.env.tmp"
            printf 'SPEECH_IMAGE=localhost/halo-ai-speech:rocm-7.14\n' \
                >>"$CONFIG_ROOT/.config.env.tmp"
            printf 'SPEECH_PORT=7860\n' >>"$CONFIG_ROOT/.config.env.tmp"
            printf 'SPEECH_TEST_AUDIO_PATH=/var/cache/halo-ai/speech/input1.wav\n' \
                >>"$CONFIG_ROOT/.config.env.tmp"
            chmod 0640 "$CONFIG_ROOT/.config.env.tmp"
            chown root:"$run_gid" "$CONFIG_ROOT/.config.env.tmp"
            mv -T "$CONFIG_ROOT/.config.env.tmp" "$CONFIG_ROOT/config.env"
        fi
    fi
    run_action chown root:"$run_gid" "$CONFIG_ROOT/config.env"
    run_action chmod 0640 "$CONFIG_ROOT/config.env"
    run_action install -m 0644 -o root -g root \
        "$source_root/config/models.d/strix-halo.json" "$CONFIG_ROOT/models.d/00-strix-halo.json"
    for preset in "$source_root"/config/request-presets.d/*.json; do
        run_action install -m 0644 -o root -g root "$preset" "$CONFIG_ROOT/request-presets.d/${preset##*/}"
    done
}

verify_links() {
    [[ -L "$CURRENT_LINK" && "$(normalize "$CURRENT_LINK")" == "$release_root" &&
       -L "$COMMAND_LINK" && "$(normalize "$COMMAND_LINK")" == "$release_root/bin/halo-ai" ]]
}

replace_link() {
    local target=$1
    local link=$2
    local temporary="${link}.halo-ai-new"
    if [[ -e "$link" && ! -L "$link" ]]; then
        die "refusing to replace non-symlink: $link"
    fi
    if [[ -e "$temporary" || -L "$temporary" ]]; then
        [[ -L "$temporary" ]] || die "refusing unexpected temporary link path: $temporary"
        run_action rm -f -- "$temporary"
    fi
    run_action ln -sfn -- "$target" "$temporary"
    run_action mv -fT -- "$temporary" "$link"
}

action_links() {
    replace_link "$release_root" "$CURRENT_LINK"
    replace_link "$CURRENT_LINK/bin/halo-ai" "$COMMAND_LINK"
}

verify_verify() {
    runuser -u "$run_user" -- env HOME="$run_home" XDG_RUNTIME_DIR="/run/user/$run_uid" \
        "$COMMAND_LINK" --config "$CONFIG_ROOT/config.env" profiles list >/dev/null
}

action_verify() {
    if "$dry_run"; then
        printf '  %q --config %q profiles list\n' "$COMMAND_LINK" "$CONFIG_ROOT/config.env"
    else
        runuser -u "$run_user" -- env HOME="$run_home" XDG_RUNTIME_DIR="/run/user/$run_uid" \
            "$COMMAND_LINK" --config "$CONFIG_ROOT/config.env" profiles list >/dev/null
    fi
}

stages=(directories release config links verify)
for stage in "${stages[@]}"; do
    declare -F "action_$stage" >/dev/null || die "installer stage $stage has no action function"
    declare -F "verify_$stage" >/dev/null || die "installer stage $stage has no verifier function"
done
case "$repair_stage" in
    ""|directories|release|config|links|verify) ;;
    *) die "unknown repair stage: $repair_stage" ;;
esac

if [[ -r "$JOURNAL" ]]; then
    journal_release=$(read_journal_value RELEASE_ID || true)
    journal_hash=$(read_journal_value SOURCE_HASH || true)
    journal_user=$(read_journal_value RUN_USER || true)
    journal_status=$(read_journal_value STATUS || true)
    journal_stages=$(read_journal_value COMPLETED_STAGES || true)
    journal_failed_stage=$(read_journal_value FAILED_STAGE || true)
    if "$restart_journal"; then
        archive_journal=true
        journal_archive="$JOURNAL_ROOT/install-state.$(date -u +%Y%m%dT%H%M%SZ).env"
        printf 'The prior journal will be archived to %s inside the confirmed Snapper transaction.\n' "$journal_archive"
    elif [[ "$journal_release" == "$release_id" && "$journal_user" == "$run_user" ]]; then
        completed_stages=$journal_stages
        printf 'Resuming release %s; completed stages: %s\n' "$release_id" "${completed_stages:-none}"
    elif [[ "$journal_status" == failed && "$journal_failed_stage" == verify &&
            "$journal_user" == "$run_user" && "$journal_release" != "$release_id" ]]; then
        journal_release_root="$RELEASES_ROOT/$journal_release"
        if [[ ",$journal_stages," == *",release,"* &&
              ",$journal_stages," == *",links,"* &&
              "$journal_release" =~ ^0\.1\.0-[0-9a-f]{16}$ &&
              "$journal_hash" =~ ^[0-9a-f]{16}$ &&
              -r "$journal_release_root/.source-hash" &&
              "$(<"$journal_release_root/.source-hash")" == "$journal_hash" &&
              -L "$COMMAND_LINK" &&
              "$(normalize "$COMMAND_LINK")" == "$journal_release_root/bin/halo-ai" ]] &&
           runuser -u "$run_user" -- env HOME="$run_home" XDG_RUNTIME_DIR="/run/user/$run_uid" \
               "$COMMAND_LINK" --config "$CONFIG_ROOT/config.env" profiles list >/dev/null; then
            release_id=$journal_release
            release_hash=$journal_hash
            release_root=$journal_release_root
            completed_stages=$journal_stages
            resuming_pinned_release=true
            printf 'The application check for failed verify stage passes; resuming release %s with corrected installer source.\n' \
                "$release_id"
        else
            die "immutable release $journal_release fails independent application verification and corrected source is $release_id; do not uninstall—rerun with --restart-journal to archive the failed attempt and deploy the corrected release"
        fi
    elif [[ "$journal_status" == failed && "$journal_user" == "$run_user" &&
            ",$journal_stages," == *",release,"* &&
            "$journal_release" =~ ^0\.1\.0-[0-9a-f]{16}$ &&
            "$journal_hash" =~ ^[0-9a-f]{16}$ &&
            -r "$RELEASES_ROOT/$journal_release/.source-hash" &&
            "$(<"$RELEASES_ROOT/$journal_release/.source-hash")" == "$journal_hash" ]]; then
        release_id=$journal_release
        release_hash=$journal_hash
        release_root="$RELEASES_ROOT/$release_id"
        completed_stages=$journal_stages
        resuming_pinned_release=true
        if [[ ",$completed_stages," == *",config,"* ]] &&
           ! runuser -u "$run_user" -- test -r "$CONFIG_ROOT/config.env"; then
            if [[ -n "$repair_stage" && "$repair_stage" != config ]]; then
                die "config.env is unreadable by $run_user and requires --repair-stage config"
            fi
            repair_stage=config
            printf 'The completed config stage has the known root:root permission defect; scheduling its guarded repair.\n'
        fi
        printf 'Resuming verified failed release %s with corrected installer source; completed stages: %s\n' \
            "$release_id" "${completed_stages:-none}"
    else
        die "journal belongs to release ${journal_release:-unknown}/user ${journal_user:-unknown}; use --restart-journal to archive and revalidate"
    fi
fi

printf 'halo-ai host install plan\n'
printf '  Source:       %s\n' "$source_root"
printf '  Release:      %s\n' "$release_root"
if "$resuming_pinned_release"; then
    printf '  Source update: deferred until resume completes (%s)\n' "$requested_release_id"
fi
printf '  Run user:     %s (UID %s)\n' "$run_user" "$run_uid"
printf '  Configuration:%s\n' " $CONFIG_ROOT"
printf '  Models:       PRESERVED (%s)\n' "$configured_model_root"
printf '  Resume journal:%s\n' " $JOURNAL"
printf '  Snapper:      %s\n' "$([[ "$use_snapper" == true ]] && echo pre/post || echo disabled)"

if "$dry_run"; then
    for stage in "${stages[@]}"; do printf '  stage: %s\n' "$stage"; done
    printf 'Dry run complete; no changes were made.\n'
    exit 0
fi

if ! "$assume_yes"; then
    [[ -t 0 ]] || die "interactive confirmation unavailable; inspect --dry-run, then use --yes"
    printf 'Type "install halo-ai" to continue: '
    IFS= read -r confirmation
    [[ "$confirmation" == "install halo-ai" ]] || die "confirmation did not match"
fi

finish_install() {
    local exit_status=$?
    local outcome=failed
    trap - EXIT
    ((exit_status == 0)) && outcome=complete
    if [[ -d "$JOURNAL_ROOT" ]]; then
        if ((exit_status == 0)); then
            write_journal complete
        else
            write_journal failed "$active_stage" || true
        fi
    fi
    if [[ -n "$pre_snapshot" && -z "$post_snapshot" ]]; then
        if post_snapshot=$(snapper -c "$SNAPSHOT_CONFIG" create --type post \
            --pre-number "$pre_snapshot" --print-number \
            --cleanup-algorithm number \
            --description "halo-ai install $outcome ($release_id)" \
            --userdata "important=yes,halo_ai=install,outcome=$outcome,release=$release_id"); then
            printf 'Snapper pair: %s -> %s (%s)\n' "$pre_snapshot" "$post_snapshot" "$outcome"
        else
            warn "could not close Snapper pre-snapshot $pre_snapshot"
        fi
    fi
    if ((exit_status != 0)); then
        printf 'Installation stopped in stage %s. Fix the reported issue, then rerun:\n' "$active_stage" >&2
        printf '  sudo %q --run-user %q --yes%s\n' "$source_root/install.sh" "$run_user" \
            "$([[ "$use_snapper" == false ]] && echo ' --no-snapshot' || true)" >&2
    fi
    exit "$exit_status"
}
trap finish_install EXIT

if "$use_snapper"; then
    command -v snapper >/dev/null || die "Snapper is missing; repair it or explicitly use --no-snapshot"
    if ! snapper_check=$(snapper -c "$SNAPSHOT_CONFIG" get-config 2>&1); then
        detected_configs=$(snapper --csvout --no-headers list-configs 2>&1 || true)
        die "Snapper config '$SNAPSHOT_CONFIG' is unavailable: $snapper_check; detected configs: ${detected_configs:-none}; repair it or explicitly use --no-snapshot"
    fi
    pre_snapshot=$(snapper -c "$SNAPSHOT_CONFIG" create --type pre --print-number \
        --cleanup-algorithm number \
        --description "halo-ai install $release_id" \
        --userdata "important=yes,halo_ai=install,release=$release_id")
    [[ "$pre_snapshot" =~ ^[0-9]+$ ]] || die "Snapper returned an invalid pre-snapshot number"
fi

if "$archive_journal"; then
    [[ "$journal_archive" == "$JOURNAL_ROOT/install-state."*.env ]] || die "unsafe journal archive path"
    [[ ! -e "$journal_archive" ]] || die "journal archive already exists: $journal_archive"
    mv -- "$JOURNAL" "$journal_archive"
    printf 'Archived prior journal to %s.\n' "$journal_archive"
fi

action_directories
mark_complete directories

for stage in "${stages[@]:1}"; do
    active_stage=$stage
    verify_function="verify_$stage"
    action_function="action_$stage"
    if stage_complete "$stage" && [[ "$repair_stage" != "$stage" ]]; then
        if "$verify_function"; then
            printf 'Stage %-11s verified; skipping.\n' "$stage"
            continue
        fi
        die "completed stage $stage no longer verifies; inspect it and rerun with --repair-stage $stage"
    fi
    printf 'Running stage %s...\n' "$stage"
    "$action_function"
    "$verify_function" || die "stage $stage did not pass verification"
    mark_complete "$stage"
done

printf 'Installed halo-ai release %s.\n' "$release_id"
if "$resuming_pinned_release"; then
    printf 'The failed release was completed without rewriting verified stages.\n'
    printf 'To deploy the newer source release afterward, rerun with --restart-journal.\n'
fi
printf 'Next: run %s doctor as %s.\n' "$COMMAND_LINK" "$run_user"
