#!/bin/sh
# Hugging Face CLI Installer for Linux/MacOS
# Usage: curl -LsSf https://hf.co/cli/install.sh | sh -s -- [OPTIONS]


if [ -z "$BASH_VERSION" ]; then
    if command -v bash >/dev/null 2>&1; then
        if [ -f "$0" ] && [ "$0" != "sh" ]; then
            exec bash "$0" "$@"
        else
            tmp_dir=$(mktemp -d 2>/dev/null || mktemp -d -t hf-cli-install)
            tmp_script="$tmp_dir/install.sh"
            cat >"$tmp_script"
            chmod +x "$tmp_script"
            bash "$tmp_script" "$@"
            status=$?
            rm -rf "$tmp_dir"
            exit $status
        fi
    else
        echo "[ERROR] bash is required to run this installer." >&2
        exit 1
    fi
fi

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

LOG_LEVEL=1

FORCE_REINSTALL="false"
BIN_DIR="${HF_CLI_BIN_DIR:-$HOME/.local/bin}"
UPDATED_RC_FILE=""
REQUESTED_VERSION="${HF_CLI_VERSION:-}"
SKIP_PATH_UPDATE="false"
UPDATED_FISH_PATH="false"
USE_UV="${HF_CLI_USE_UV:-true}"
UV_INSTALLED="false"

log_debug() {
    if [ "$LOG_LEVEL" -lt 2 ]; then
        return 0
    fi
    printf '%b\n' "${BLUE}[DEBUG]${NC} $1"
}

log_info() {
    if [ "$LOG_LEVEL" -lt 1 ]; then
        return 0
    fi
    printf '%b\n' "${BLUE}[INFO]${NC} $1"
}

log_success() {
    printf '%b\n' "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    printf '%b\n' "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    printf '%b\n' "${RED}[ERROR]${NC} $1" >&2
}

run_command() {
    local description="$1"
    shift
    set +e
    "$@"
    local status=$?
    set -e
    if [ $status -ne 0 ]; then
        log_error "$description"
        log_error "Command: $*"
        log_error "Re-run with --verbose for detailed output."
        exit $status
    fi
}

usage() {
    cat <<'EOF'
Usage: curl -LsSf https://hf.co/cli/install.sh | sh -s -- [OPTIONS]

Options:
  --force           Recreate the Hugging Face CLI virtual environment if it exists
  --no-modify-path  Skip adding the hf wrapper directory to PATH
  --no-uv           Use pip instead of uv for package installation
  -v, --verbose     Enable verbose output
  --help, -h        Show this message and exit

Environment variables:
  HF_HOME           Installation base directory; installer uses $HF_HOME/cli when set
  HF_CLI_BIN_DIR    Directory for the hf wrapper (default: ~/.local/bin)
  HF_CLI_VERSION    Install a specific huggingface_hub version (default: latest)
  HF_CLI_USE_UV     Use uv for package installation (default: true)
EOF
}

expand_path() {
    local input="$1"
    if [ -z "$input" ]; then
        return 0
    fi

    case "$input" in
        ~)
            printf '%s\n' "$HOME"
            ;;
        ~/*)
            printf '%s/%s\n' "$HOME" "${input#~/}"
            ;;
        /*)
            printf '%s\n' "$input"
            ;;
        *)
            printf '%s/%s\n' "$PWD" "$input"
            ;;
    esac
}

while [ $# -gt 0 ]; do
    case "$1" in
        --force)
            FORCE_REINSTALL="true"
            ;;
        --no-modify-path)
            SKIP_PATH_UPDATE="true"
            ;;
        --no-uv)
            USE_UV="false"
            ;;
        -v|--verbose)
            LOG_LEVEL=2
            HF_CLI_VERBOSE_PIP=1
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

BIN_DIR=$(expand_path "$BIN_DIR")

if [ -n "$HF_HOME" ]; then
    HF_CLI_DIR="$HF_HOME/cli"
else
    HF_CLI_DIR="$HOME/.hf-cli"
fi

HF_CLI_DIR=$(expand_path "$HF_CLI_DIR")
VENV_DIR="$HF_CLI_DIR/venv"

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

python_version_supported() {
    "$1" <<'PY' >/dev/null 2>&1
import sys
sys.exit(0 if sys.version_info >= (3, 9) else 1)
PY
}

ensure_python() {
    local candidates=(python3 python)
    local chosen=""
    local version_output=""

    for candidate in "${candidates[@]}"; do
        if command_exists "$candidate"; then
            version_output="$($candidate --version 2>&1)"
            if python_version_supported "$candidate"; then
                PYTHON_CMD="$candidate"
                chosen="$candidate"
                break
            else
                log_warning "$candidate detected ($version_output) but Python 3.9+ is required."
            fi
        fi
    done

    if [ -z "$chosen" ]; then
        log_error "Python 3.9+ is required but was not found."
        case "$(detect_os)" in
            macos)
                log_info "On macOS: brew install python (or download Python 3.9+ from python.org)"
                ;;
            linux)
                if command_exists apt-get || command_exists apt; then
                    log_info "On Debian/Ubuntu: sudo apt update && sudo apt install python3 python3-pip python3-venv"
                elif command_exists dnf; then
                    log_info "On Fedora/RHEL: sudo dnf install python3 python3-pip"
                elif command_exists yum; then
                    log_info "On CentOS/RHEL: sudo yum install python3 python3-pip"
                else
                    log_info "Install Python 3.9+ with your distro's package manager."
                fi
                ;;
            *)
                log_info "Install Python 3.9+ from https://www.python.org/downloads/"
                ;;
        esac
        exit 1
    fi

    log_info "Using Python: $($PYTHON_CMD --version)"
}

ensure_uv() {
    if [ "$USE_UV" != "true" ]; then
        log_info "Using pip for package installation"
        return
    fi
    
    if command_exists uv; then
        UV_CMD="uv"
        log_info "Using uv: $(uv --version 2>&1 || echo 'version unknown')"
        return
    fi

    log_info "Installing uv package manager..."
    
    local uv_installer=$(mktemp)
    if command_exists curl; then
        if ! curl -LsSf https://astral.sh/uv/install.sh -o "$uv_installer" 2>/dev/null; then
            log_warning "Failed to download uv installer, falling back to pip"
            USE_UV="false"
            rm -f "$uv_installer"
            return
        fi
    elif command_exists wget; then
        if ! wget -qO "$uv_installer" https://astral.sh/uv/install.sh 2>/dev/null; then
            log_warning "Failed to download uv installer, falling back to pip"
            USE_UV="false"
            rm -f "$uv_installer"
            return
        fi
    else
        log_warning "curl or wget not found, falling back to pip"
        USE_UV="false"
        return
    fi
    
    chmod +x "$uv_installer"
    
    export INSTALLER_NO_MODIFY_PATH=1
    if ! sh "$uv_installer" >/dev/null 2>&1; then
        log_warning "Failed to install uv, falling back to pip"
        USE_UV="false"
        rm -f "$uv_installer"
        return
    fi
    rm -f "$uv_installer"
    
    local uv_locations=(
        "$HOME/.cargo/bin/uv"
        "$HOME/.local/bin/uv"
        "/usr/local/bin/uv"
    )
    
    for location in "${uv_locations[@]}"; do
        if [ -x "$location" ]; then
            UV_CMD="$location"
            UV_INSTALLED="true"
            log_info "uv installed: $($UV_CMD --version)"
            return
        fi
    done
    
    log_warning "uv installation completed but command not found, falling back to pip"
    USE_UV="false"
}

create_directories() {
    log_info "Creating directories..."
    run_command "Failed to create install directory $HF_CLI_DIR" mkdir -p "$HF_CLI_DIR"
    run_command "Failed to create bin directory $BIN_DIR" mkdir -p "$BIN_DIR"
}

create_venv() {
    log_info "Creating virtual environment..."
    if [ -d "$VENV_DIR" ]; then
        if [ "$FORCE_REINSTALL" = "true" ]; then
            log_warning "Virtual environment already exists; removing it since --force was passed"
            rm -rf "$VENV_DIR"
        else
            log_info "Virtual environment already exists; reusing (pass --force to recreate)"
            return
        fi
    fi

    if [ "$USE_UV" = "true" ]; then
        local uv_args=()
        if [ "$LOG_LEVEL" -ge 2 ]; then
            uv_args+=(--verbose)
        else
            uv_args+=(--quiet)
        fi

        run_command "Failed to create virtual environment at $VENV_DIR" "$UV_CMD" venv "${uv_args[@]}" --python "$PYTHON_CMD" "$VENV_DIR"
    else
        if ! "$PYTHON_CMD" -m venv --help >/dev/null 2>&1; then
            log_error "Python's venv module is unavailable. Install python3-venv and retry."
            case "$(detect_os)" in
                linux)
                    log_info "Try: sudo apt install python3-venv      # Debian/Ubuntu"
                    log_info "     sudo dnf install python3-venv      # Fedora"
                    ;;
                macos)
                    log_info "Try reinstalling Python via Homebrew: brew install python"
                    ;;
            esac
            exit 1
        fi

        run_command "Failed to create virtual environment at $VENV_DIR" "$PYTHON_CMD" -m venv "$VENV_DIR"
        
        log_info "Upgrading pip..."
        run_command "Failed to upgrade pip" "$VENV_DIR/bin/python" -m pip install --upgrade pip
    fi

    touch "$VENV_DIR/.hf_installer_marker"
}

install_hf_hub() {
    local package_spec="huggingface_hub"
    if [ -n "$REQUESTED_VERSION" ]; then
        package_spec="huggingface_hub==$REQUESTED_VERSION"
        log_info "Installing The Hugging Face CLI (version $REQUESTED_VERSION)..."
    else
        log_info "Installing The Hugging Face CLI (latest)..."
    fi

    if [ "$USE_UV" = "true" ]; then
        local -a uv_flags=()
        if [ "${HF_CLI_VERBOSE_PIP:-}" != "1" ]; then
            uv_flags+=(--quiet)
        fi

        run_command "Failed to install $package_spec" "$UV_CMD" pip install --python "$VENV_DIR/bin/python" "${uv_flags[@]}" "$package_spec"
    else
        local extra_pip_args="${HF_CLI_PIP_ARGS:-${HF_PIP_ARGS:-}}"
        local -a pip_flags
        if [ "${HF_CLI_VERBOSE_PIP:-}" = "1" ]; then
            pip_flags=()
        else
            pip_flags=(--quiet --progress-bar off --disable-pip-version-check)
        fi

        if [ -n "$extra_pip_args" ]; then
            log_info "Passing extra pip arguments: $extra_pip_args"
        fi

        if [ -n "$extra_pip_args" ]; then
            # shellcheck disable=SC2086
            run_command "Failed to install $package_spec" "$VENV_DIR/bin/python" -m pip install --upgrade "$package_spec" ${pip_flags[*]} $extra_pip_args
        else
            # shellcheck disable=SC2086
            run_command "Failed to install $package_spec" "$VENV_DIR/bin/python" -m pip install --upgrade "$package_spec" ${pip_flags[*]}
        fi
    fi
}

expose_cli_command() {
    log_info "Linking hf CLI into $BIN_DIR..."

    local source_cli="$VENV_DIR/bin/hf"
    if [ ! -x "$source_cli" ]; then
        log_error "hf command not found in the virtual environment at $source_cli"
        log_error "Verify that The Hugging Face CLI is installed correctly."
        exit 1
    fi

    local link_method=""
    if ln -sf "$source_cli" "$BIN_DIR/hf" 2>/dev/null; then
        link_method="symlink"
    else
        if cp "$source_cli" "$BIN_DIR/hf" 2>/dev/null; then
            link_method="copy"
        else
            log_error "Failed to place hf command in $BIN_DIR (tried symlink and copy)."
            exit 1
        fi
    fi

    chmod +x "$BIN_DIR/hf"

    if [ "$link_method" = "symlink" ]; then
        log_info "hf available at $BIN_DIR/hf (symlink to venv)"
    else
        log_info "hf available at $BIN_DIR/hf"
    fi
    log_info "Run without touching PATH: env PATH=\"$BIN_DIR:\$PATH\" hf --help"
}

update_path() {
    local shell_rc=""
    local -a shell_rc_candidates=()

    case "$SHELL" in
        */bash)
            shell_rc_candidates=()
            shell_rc_candidates+=("$HOME/.bashrc")
            shell_rc_candidates+=("$HOME/.bash_profile")
            shell_rc_candidates+=("$HOME/.profile")
            ;;
        */zsh)
            shell_rc_candidates=()
            shell_rc_candidates+=("$HOME/.zshrc")
            shell_rc_candidates+=("$HOME/.zprofile")
            ;;
        */fish)
            shell_rc_candidates=()
            if command -v fish >/dev/null 2>&1; then
                if fish -c "contains \"$BIN_DIR\" \$fish_user_paths" >/dev/null 2>&1; then
                    log_info "$BIN_DIR already present in fish_user_paths"
                    UPDATED_FISH_PATH="true"
                    return
                elif fish -c "set -Ux fish_user_paths \"$BIN_DIR\" \$fish_user_paths" >/dev/null 2>&1; then
                    UPDATED_FISH_PATH="true"
                    log_success "Added $BIN_DIR to fish_user_paths"
                    log_info "Apply it now with: set -Ux fish_user_paths $BIN_DIR \$fish_user_paths"
                    return
                else
                    log_warning "Could not update fish_user_paths automatically."
                fi
            fi
            ;;
        *)
            shell_rc_candidates=()
            shell_rc_candidates+=("$HOME/.profile")
            ;;
    esac

    if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
        if [ "$SKIP_PATH_UPDATE" = "true" ]; then
            log_info "Skipping PATH update (--no-modify-path)."
            return
        fi

        log_info "$BIN_DIR is not in your PATH"

            if [ "${#shell_rc_candidates[@]}" -gt 0 ]; then
                for candidate in "${shell_rc_candidates[@]}"; do
                    if [ -f "$candidate" ]; then
                        shell_rc="$candidate"
                        break
                    fi
            done

            if [ -z "$shell_rc" ]; then
                shell_rc="${shell_rc_candidates[0]}"
                log_info "Creating shell config file at $shell_rc to update PATH"
                touch "$shell_rc"
            fi

            if ! grep -Fq "$BIN_DIR" "$shell_rc"; then
                {
                    echo ""
                    echo "# Added by Hugging Face CLI installer"
                    echo "export PATH=\"$BIN_DIR:\$PATH\""
                } >> "$shell_rc"
                UPDATED_RC_FILE="$shell_rc"
                log_success "Added $BIN_DIR to PATH via $shell_rc"
                if [ "$LOG_LEVEL" -ge 1 ]; then
                    log_info "Apply it now with: source $shell_rc"
                fi
            fi
        else
            log_warning "Could not automatically update PATH for your shell."
            if [[ "$SHELL" == *"/fish" ]]; then
                if [ "$UPDATED_FISH_PATH" != "true" ]; then
                    log_warning "Run: set -Ux fish_user_paths $BIN_DIR \$fish_user_paths"
                fi
            else
                log_warning "Add this line to your shell config: export PATH=\"$BIN_DIR:\$PATH\""
            fi
        fi
    fi
}

verify_installation() {
    log_info "Verifying installation..."
    
    if [ -x "$BIN_DIR/hf" ]; then
        if "$BIN_DIR/hf" version >/dev/null 2>&1; then
            log_success "Hugging Face CLI installed successfully!"
            log_info "CLI location: $BIN_DIR/hf"
            log_info "Installation directory: $HF_CLI_DIR"
        else
            log_error "Installation verification failed. The hf command is not working properly."
            exit 1
        fi
    else
        log_error "Installation failed. Wrapper script not found."
        exit 1
    fi
}

show_uninstall_info() {
    log_info ""
    log_info "To uninstall the Hugging Face CLI, run:"
    log_info "  rm -rf $HF_CLI_DIR"
    log_info "  rm -f $BIN_DIR/hf"
    log_info ""
    if [ -n "$UPDATED_RC_FILE" ]; then
        log_info "  (shell) Undo PATH entry: sed -i.bak '/Added by Hugging Face CLI installer/d' $UPDATED_RC_FILE && rm -f ${UPDATED_RC_FILE}.bak"
    elif [ "$UPDATED_FISH_PATH" = "true" ]; then
        log_info "  (fish) Undo PATH entry: fish -c 'set -Ux fish_user_paths (string match -v \"$BIN_DIR\" \$fish_user_paths)'"
    elif [ "$SKIP_PATH_UPDATE" = "true" ]; then
        log_info "  (PATH unchanged because --no-modify-path was used)"
    else
        log_info "  Remove any PATH edits you made manually."
    fi
}

main() {
    log_info "Installing Hugging Face CLI..."
    log_info "OS: $(detect_os)"
    log_info "Force reinstall: $FORCE_REINSTALL"
    log_info "Install dir: $HF_CLI_DIR"
    log_info "Bin dir: $BIN_DIR"
    log_info "Requested version: ${REQUESTED_VERSION:-latest}"
    log_info "Skip PATH update: $SKIP_PATH_UPDATE"

    ensure_python
    ensure_uv
    create_directories
    create_venv
    install_hf_hub
    expose_cli_command
    update_path
    verify_installation

    if [[ ":$PATH:" == *":$BIN_DIR:"* ]]; then
        log_info "Current version: $(hf version 2>/dev/null || echo 'Run source ~/.bashrc or restart your shell first')"
    else
        log_info "Current version: $($BIN_DIR/hf version)"
    fi

    show_uninstall_info

    log_success "hf CLI ready!"
    log_info "Binary: $BIN_DIR/hf"
    log_info "Virtualenv: $HF_CLI_DIR"
    log_info "CLI version: ${REQUESTED_VERSION:-latest}"
    log_info "Try it now: env PATH=\"$BIN_DIR:\$PATH\" hf --help"
    log_info "Examples:"
    log_info "  hf login"
    log_info "  hf download deepseek-ai/DeepSeek-R1"
    log_info "  hf jobs run python:3.12 python -c 'print(\"Hello from HF CLI!\")'"
    log_info ""
}

trap 'log_error "Installation interrupted"; exit 130' INT

main "$@"
