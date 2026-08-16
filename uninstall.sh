#!/usr/bin/env bash

set -Eeuo pipefail

readonly INSTALL_ROOT="/opt/halo-ai"
readonly CONFIG_ROOT="/etc/opt/halo-ai"
readonly STATE_ROOT="/var/opt/halo-ai"
readonly CACHE_ROOT="/var/cache/halo-ai"
readonly INSTALLER_STATE_ROOT="/var/lib/halo-ai-installer"
readonly COMMAND_LINK="/usr/local/bin/halo-ai"
readonly EXTERNAL_MODEL_ROOT="/srv/halo-ai/models"
readonly MANAGED_LABEL="local.halo-ai.managed=true"
readonly SNAPSHOT_CONFIG="root"

dry_run=false
assume_yes=false
keep_podman=false
keep_config=false
keep_state=false
keep_cache=false
remove_images=false
use_snapper=true
run_user=""

usage() {
    cat <<'EOF'
Usage: sudo ./uninstall.sh [options]

Remove the installed halo-ai runtime without modifying /srv/halo-ai/models.

Options:
  --run-user USER   Owner of the rootless Podman store (normally your login)
  --keep-podman     Keep halo-ai containers and volumes
  --keep-config     Keep /etc/opt/halo-ai
  --keep-state      Keep /var/opt/halo-ai
  --keep-cache      Keep /var/cache/halo-ai
  --remove-images   Also remove images labeled local.halo-ai.managed=true
  --no-snapshot     Explicitly uninstall without a Snapper pre/post pair
  --dry-run         Show the resolved plan without changing anything
  --yes             Skip the interactive confirmation
  -h, --help        Show this help

The external model root is an immutable exclusion: no option removes, scans,
changes ownership of, or otherwise modifies /srv/halo-ai/models.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

warn() {
    printf 'warning: %s\n' "$*" >&2
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

run_command() {
    if "$dry_run"; then
        print_command "$@"
    else
        "$@"
    fi
}

while (($#)); do
    case "$1" in
        --run-user)
            (($# >= 2)) || die "--run-user requires an argument"
            run_user=$2
            shift 2
            ;;
        --keep-podman)
            keep_podman=true
            shift
            ;;
        --keep-config)
            keep_config=true
            shift
            ;;
        --keep-state)
            keep_state=true
            shift
            ;;
        --keep-cache)
            keep_cache=true
            shift
            ;;
        --remove-images)
            remove_images=true
            shift
            ;;
        --no-snapshot)
            use_snapper=false
            shift
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        --yes)
            assume_yes=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

if "$keep_podman" && "$remove_images"; then
    die "--keep-podman and --remove-images cannot be used together"
fi

if ! "$dry_run" && ((EUID != 0)); then
    die "run the uninstaller as root (normally: sudo ./uninstall.sh)"
fi

read_config_value() {
    local key=$1
    local line value=""

    [[ -r "$CONFIG_ROOT/config.env" ]] || return 1
    [[ "$key" =~ ^[A-Z0-9_]+$ ]] || return 1

    while IFS= read -r line || [[ -n "$line" ]]; do
        if [[ "$line" =~ ^[[:space:]]*${key}[[:space:]]*=[[:space:]]*(.*)$ ]]; then
            value=${BASH_REMATCH[1]}
        fi
    done <"$CONFIG_ROOT/config.env"
    [[ -n "$value" ]] || return 1

    if [[ "$value" == \"*\" && "$value" == *\" ]]; then
        value=${value:1:${#value}-2}
    elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
        value=${value:1:${#value}-2}
    fi
    printf '%s\n' "$value"
}

if [[ -z "$run_user" ]]; then
    run_user=$(read_config_value HALO_AI_RUN_USER || true)
fi
if [[ -z "$run_user" && -n "${SUDO_USER:-}" && "$SUDO_USER" != "root" ]]; then
    run_user=$SUDO_USER
fi
if [[ -z "$run_user" ]] && ((EUID != 0)); then
    run_user=$(id -un)
fi
[[ -n "$run_user" ]] || die \
    "cannot determine the rootless Podman owner; pass --run-user USER"
[[ "$run_user" =~ ^[a-zA-Z0-9_.-]+$ ]] || die "invalid run user: $run_user"
[[ "$run_user" != "root" ]] || die "the Podman run user must not be root"

passwd_entry=$(getent passwd "$run_user") || die "unknown run user: $run_user"
IFS=: read -r _ _ run_uid _ _ run_home _ <<<"$passwd_entry"
[[ "$run_uid" =~ ^[0-9]+$ ]] || die "invalid UID for run user: $run_user"
[[ -d "$run_home" ]] || die "home directory does not exist for $run_user: $run_home"

run_as_user() (
    cd -- "$run_home"
    if ((EUID == run_uid)); then
        env HOME="$run_home" XDG_RUNTIME_DIR="/run/user/$run_uid" "$@"
    else
        runuser -u "$run_user" -- \
            env HOME="$run_home" XDG_RUNTIME_DIR="/run/user/$run_uid" "$@"
    fi
)

run_as_user_command() {
    if "$dry_run"; then
        if ((EUID == run_uid)); then
            print_command env HOME="$run_home" \
                XDG_RUNTIME_DIR="/run/user/$run_uid" "$@"
        else
            print_command runuser -u "$run_user" -- env HOME="$run_home" \
                XDG_RUNTIME_DIR="/run/user/$run_uid" "$@"
        fi
    else
        run_as_user "$@"
    fi
}

normalize_path() {
    readlink -m -- "$1"
}

validate_removal_target() {
    local target=$1
    local expected=$2
    local normalized expected_normalized

    normalized=$(normalize_path "$target")
    expected_normalized=$(normalize_path "$expected")
    [[ "$normalized" == "$expected_normalized" ]] || die \
        "refusing unexpected removal target: $target resolved to $normalized"
    [[ "$normalized" == /* && "$normalized" != "/" ]] || die \
        "refusing unsafe removal target: $normalized"

    local protected protected_normalized
    for protected in "${protected_model_roots[@]}"; do
        protected_normalized=$(normalize_path "$protected")
        # Refuse overlap in either direction. This also protects a customized
        # external model path read from the installed configuration.
        if [[ "$protected_normalized" == "$normalized" ||
              "$protected_normalized" == "$normalized/"* ||
              "$normalized" == "$protected_normalized/"* ]]; then
            die "refusing removal target $normalized because it overlaps protected model root $protected"
        fi
    done
}

configured_model_root=$(read_config_value HALO_AI_MODELS_ROOT || true)
if [[ -n "$configured_model_root" ]]; then
    [[ "$configured_model_root" == /* && "$configured_model_root" != "/" ]] || die \
        "configured HALO_AI_MODELS_ROOT is not a safe absolute path: $configured_model_root"
fi
protected_model_roots=("$EXTERNAL_MODEL_ROOT")
if [[ -n "$configured_model_root" &&
      "$(normalize_path "$configured_model_root")" != "$(normalize_path "$EXTERNAL_MODEL_ROOT")" ]]; then
    protected_model_roots+=("$configured_model_root")
fi

validate_removal_target "$INSTALL_ROOT" "/opt/halo-ai"
validate_removal_target "$CONFIG_ROOT" "/etc/opt/halo-ai"
validate_removal_target "$STATE_ROOT" "/var/opt/halo-ai"
validate_removal_target "$CACHE_ROOT" "/var/cache/halo-ai"
validate_removal_target "$INSTALLER_STATE_ROOT" "/var/lib/halo-ai-installer"

check_for_nested_mounts() {
    local target=$1
    local normalized mount_target mount_normalized

    [[ -e "$target" || -L "$target" ]] || return 0
    command -v findmnt >/dev/null || die "findmnt is required for mount safety checks"
    normalized=$(normalize_path "$target")
    while IFS= read -r mount_target; do
        [[ -n "$mount_target" ]] || continue
        mount_normalized=$(normalize_path "$mount_target")
        if [[ "$mount_normalized" == "$normalized" ||
              "$mount_normalized" == "$normalized/"* ]]; then
            die "refusing removal target $target because it contains mountpoint $mount_target"
        fi
    done < <(findmnt -R -T "$target" -nro TARGET)
}

check_for_nested_mounts "$INSTALL_ROOT"
"$keep_config" || check_for_nested_mounts "$CONFIG_ROOT"
"$keep_state" || check_for_nested_mounts "$STATE_ROOT"
"$keep_cache" || check_for_nested_mounts "$CACHE_ROOT"
check_for_nested_mounts "$INSTALLER_STATE_ROOT"

if [[ -e "$COMMAND_LINK" || -L "$COMMAND_LINK" ]]; then
    [[ -L "$COMMAND_LINK" ]] || die \
        "$COMMAND_LINK is not a symlink; refusing to remove an unowned file"
    link_target=$(normalize_path "$COMMAND_LINK")
    [[ "$link_target" == "$INSTALL_ROOT" ||
       "$link_target" == "$INSTALL_ROOT/"* ]] || die \
        "$COMMAND_LINK points outside $INSTALL_ROOT; refusing to remove it"
fi

find_model_artifact() {
    local target=$1
    [[ -d "$target" ]] || return 1
    find -P "$target" -xdev -type f \
        \( -iname '*.gguf' -o -iname '*.ggml' -o -iname '*.safetensors' \
        -o -iname '*.onnx' -o -iname '*.pth' -o -iname '*.pt' \) \
        -print -quit
}

check_tree_for_misplaced_models() {
    local target=$1
    local artifact

    artifact=$(find_model_artifact "$target" || true)
    [[ -z "$artifact" ]] || die \
        "model artifact found under removal target $target: $artifact; move it outside the installed solution before uninstalling"
}

check_tree_for_misplaced_models "$INSTALL_ROOT"
"$keep_config" || check_tree_for_misplaced_models "$CONFIG_ROOT"
"$keep_state" || check_tree_for_misplaced_models "$STATE_ROOT"
"$keep_cache" || check_tree_for_misplaced_models "$CACHE_ROOT"

managed_containers=()
managed_volumes=()
managed_images=()
legacy_volumes=()

if ! "$keep_podman"; then
    command -v podman >/dev/null || die \
        "podman is unavailable; install it or pass --keep-podman"
    if ((EUID != run_uid)); then
        command -v runuser >/dev/null || die "runuser is required"
    fi
    [[ -d "/run/user/$run_uid" ]] || die \
        "the runtime directory /run/user/$run_uid is absent; log in as $run_user or pass --keep-podman"
    run_as_user podman info >/dev/null || die \
        "cannot access the rootless Podman store for $run_user"

    mapfile -t managed_containers < <(
        run_as_user podman ps -aq --filter "label=$MANAGED_LABEL"
    )
    mapfile -t managed_volumes < <(
        run_as_user podman volume ls -q --filter "label=$MANAGED_LABEL"
    )
    if "$remove_images"; then
        mapfile -t managed_images < <(
            run_as_user podman images -q --filter "label=$MANAGED_LABEL" | sort -u
        )
    fi

    for volume in \
        halo-lemonade-huggingface halo-lemonade-llama halo-lemonade-config; do
        if run_as_user podman volume exists "$volume"; then
            if ! printf '%s\n' "${managed_volumes[@]}" | grep -Fxq -- "$volume"; then
                legacy_volumes+=("$volume")
            fi
        fi
    done
fi

covered_change=false
[[ -e "$INSTALL_ROOT" || -L "$INSTALL_ROOT" ]] && covered_change=true
[[ -e "$COMMAND_LINK" || -L "$COMMAND_LINK" ]] && covered_change=true
if ! "$keep_config" && [[ -e "$CONFIG_ROOT" || -L "$CONFIG_ROOT" ]]; then
    covered_change=true
fi
if ! "$keep_state" && [[ -e "$STATE_ROOT" || -L "$STATE_ROOT" ]]; then
    covered_change=true
fi
if [[ -e "$INSTALLER_STATE_ROOT" || -L "$INSTALLER_STATE_ROOT" ]]; then
    covered_change=true
fi

if "$use_snapper" && "$covered_change" && ! "$dry_run"; then
    command -v snapper >/dev/null || die \
        "Snapper is unavailable; repair it or explicitly pass --no-snapshot"
    if ! snapper_check=$(snapper -c "$SNAPSHOT_CONFIG" get-config 2>&1); then
        detected_configs=$(snapper --csvout --no-headers list-configs 2>&1 || true)
        die "Snapper config '$SNAPSHOT_CONFIG' is unavailable: $snapper_check; detected configs: ${detected_configs:-none}; repair it or explicitly pass --no-snapshot"
    fi
fi

printf 'halo-ai uninstall plan\n'
printf '  Podman run user:       %s (UID %s)\n' "$run_user" "$run_uid"
printf '  Remove install root:   %s\n' "$INSTALL_ROOT"
printf '  Remove command link:   %s\n' "$COMMAND_LINK"
printf '  Remove config:         %s\n' "$([[ "$keep_config" == false ]] && echo yes || echo no)"
printf '  Remove state:          %s\n' "$([[ "$keep_state" == false ]] && echo yes || echo no)"
printf '  Remove cache:          %s\n' "$([[ "$keep_cache" == false ]] && echo yes || echo no)"
printf '  Remove install journal:%s\n' " yes ($INSTALLER_STATE_ROOT)"
printf '  Managed containers:    %s\n' "${#managed_containers[@]}"
printf '  Managed volumes:       %s\n' "${#managed_volumes[@]}"
printf '  Managed images:        %s\n' \
    "$([[ "$remove_images" == true ]] && echo "${#managed_images[@]}" || echo 'kept')"
printf '  Snapper pre/post pair: %s\n' \
    "$([[ "$use_snapper" == true && "$covered_change" == true ]] && echo yes || echo no)"
printf '  External models:       PRESERVED'
printf ' (%s)' "${protected_model_roots[@]}"
printf '\n'

if ((${#legacy_volumes[@]})); then
    warn "unlabeled legacy volumes will be kept: ${legacy_volumes[*]}"
fi

if "$dry_run"; then
    printf 'Dry run complete; no changes were made.\n'
    exit 0
fi

if ! "$assume_yes"; then
    [[ -t 0 ]] || die "confirmation requires a terminal; inspect --dry-run, then pass --yes"
    printf 'Type "uninstall halo-ai" to continue: '
    IFS= read -r confirmation
    [[ "$confirmation" == "uninstall halo-ai" ]] || die "confirmation did not match"
fi

pre_snapshot=""
post_snapshot=""
finish_snapshot_pair() {
    local exit_status=$?
    local outcome="failed"

    trap - EXIT
    if [[ -n "$pre_snapshot" && -z "$post_snapshot" ]]; then
        ((exit_status == 0)) && outcome="complete"
        if post_snapshot=$(snapper -c "$SNAPSHOT_CONFIG" create \
            --type post \
            --pre-number "$pre_snapshot" \
            --print-number \
            --cleanup-algorithm number \
            --description "halo-ai uninstall $outcome" \
            --userdata "important=yes,halo_ai=uninstall,outcome=$outcome"); then
            printf 'Created Snapper post-snapshot %s (%s).\n' \
                "$post_snapshot" "$outcome"
        else
            warn "could not close Snapper pre-snapshot $pre_snapshot with a post-snapshot"
        fi
    fi
    exit "$exit_status"
}
trap finish_snapshot_pair EXIT

if ! "$keep_podman"; then
    if ((${#managed_containers[@]})); then
        run_as_user_command podman stop --time 120 "${managed_containers[@]}"
        run_as_user_command podman rm "${managed_containers[@]}"
    fi
    if ((${#managed_volumes[@]})); then
        run_as_user_command podman volume rm "${managed_volumes[@]}"
    fi
    if "$remove_images" && ((${#managed_images[@]})); then
        run_as_user_command podman image rm "${managed_images[@]}"
    fi
fi

if "$use_snapper" && "$covered_change"; then
    pre_snapshot=$(snapper -c "$SNAPSHOT_CONFIG" create \
        --type pre \
        --print-number \
        --cleanup-algorithm number \
        --description "halo-ai uninstall" \
        --userdata "important=yes,halo_ai=uninstall")
    [[ "$pre_snapshot" =~ ^[0-9]+$ ]] || die \
        "Snapper did not return a valid pre-snapshot number"
    printf 'Created Snapper pre-snapshot %s.\n' "$pre_snapshot"
fi

if [[ -L "$COMMAND_LINK" ]]; then
    run_command rm -f -- "$COMMAND_LINK"
fi

remove_tree() {
    local target=$1
    [[ -e "$target" || -L "$target" ]] || return 0
    run_command rm -rf --one-file-system -- "$target"
}

remove_tree "$INSTALL_ROOT"
"$keep_config" || remove_tree "$CONFIG_ROOT"
"$keep_state" || remove_tree "$STATE_ROOT"
"$keep_cache" || remove_tree "$CACHE_ROOT"
remove_tree "$INSTALLER_STATE_ROOT"

if [[ -n "$pre_snapshot" ]]; then
    post_snapshot=$(snapper -c "$SNAPSHOT_CONFIG" create \
        --type post \
        --pre-number "$pre_snapshot" \
        --print-number \
        --cleanup-algorithm number \
        --description "halo-ai uninstall complete" \
        --userdata "important=yes,halo_ai=uninstall,outcome=complete")
    printf 'Created Snapper post-snapshot %s.\n' "$post_snapshot"
fi

printf 'halo-ai has been uninstalled.\n'
printf 'External model roots were preserved:'
printf ' %s' "${protected_model_roots[@]}"
printf '.\n'
printf 'The source checkout and unlabeled/shared Podman images were not modified.\n'
