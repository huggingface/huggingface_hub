#!/bin/sh
# Hugging Face CLI Installer for Linux/MacOS
# Usage: curl -LsSf https://hf.co/cli/install.sh | bash -s
# Usage: curl -LsSf https://hf.co/cli/install.sh | bash -s --with-transformers
# Usage: curl -LsSf https://hf.co/cli/install.sh | bash -s -- [OPTIONS]


if [ -z "$BASH_VERSION" ]; then
    if command -v bash >/dev/null 2>&1; then
        if [ -f "$0" ] && [ "$0" != "sh" ] && [ "$0" != "bash" ]; then
            exec bash "$0" "$@"
        else
            tmp_dir=$(mktemp -d 2>/dev/null || mktemp -d -t hf-cli-install)
            tmp_script="$tmp_dir/install.sh"
            cat >"$tmp_script"
            chmod +x "$tmp_script"
            bash "$tmp_script" "$@"
            exit_code=$?
            rm -rf "$tmp_dir"
            exit $exit_code
        fi
    else
        echo "[ERROR] bash is required to run this installer." >&2
        echo "[ERROR] Please run: curl -LsSf https://hf.co/cli/install.sh | bash" >&2
        exit 1
    fi
fi

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging level: 0=quiet, 1=info (default), 2=verbose
LOG_LEVEL=1

# Configuration
FORCE_REINSTALL="false"
BIN_DIR="${HF_CLI_BIN_DIR:-$HOME/.local/bin}"
UPDATED_RC_FILE=""
SKIP_PATH_UPDATE="false"
UPDATED_FISH_PATH="false"
WITH_TRANSFORMERS="false"

# Logging functions
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
Usage: curl -LsSf https://hf.co/cli/install.sh | bash -s -- [OPTIONS]

Options:
  --force             Recreate the Hugging Face CLI virtual environment if it exists
  --no-modify-path    Skip adding the hf wrapper directory to PATH
  --with-transformers Also install the transformers CLI
  -v, --verbose       Enable verbose output (includes full pip logs)
  --help, -h          Show this message and exit

Environment variables:
  HF_HOME           Installation base directory; installer uses $HF_HOME/cli when set
  HF_CLI_BIN_DIR    Directory for the hf wrapper (default: ~/.local/bin)
EOF
}

# Normalize user paths to absolute paths
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
        --with-transformers)
            WITH_TRANSFORMERS="true"
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

# Persist fully resolved paths for downstream use and wrapper creation
BIN_DIR=$(expand_path "$BIN_DIR")

if [ -n "$HF_HOME" ]; then
    HF_CLI_DIR="$HF_HOME/cli"
else
    HF_CLI_DIR="$HOME/.hf-cli"
fi

HF_CLI_DIR=$(expand_path "$HF_CLI_DIR")
VENV_DIR="$HF_CLI_DIR/venv"

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Install Python if not available
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
                    log_info "On Debian/Ubuntu: sudo apt update && sudo apt install python3 python3-pip"
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

# Create directories
create_directories() {
    log_info "Creating directories..."
    run_command "Failed to create install directory $HF_CLI_DIR" mkdir -p "$HF_CLI_DIR"
    run_command "Failed to create bin directory $BIN_DIR" mkdir -p "$BIN_DIR"
}

# Create virtual environment
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

    # Fail early with guidance when python lacks the venv module
    if ! "$PYTHON_CMD" -m venv --help >/dev/null 2>&1; then
        log_error "Python's venv module is unavailable. Install python3-venv / ensurepip and retry."
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

    # Mark this installation as installer-managed
    touch "$VENV_DIR/.hf_installer_marker"

    # Use the venv python for pip management
    log_info "Upgrading pip..."
    run_command "Failed to upgrade pip" "$VENV_DIR/bin/python" -m pip install --upgrade pip
}

# Install a Python package in the venv
install_package() {
    local package_spec="$1"

    local extra_pip_args="${HF_CLI_PIP_ARGS:-${HF_PIP_ARGS:-}}"
    local verbose="${HF_CLI_VERBOSE_PIP:-}"

    if [ -n "$extra_pip_args" ]; then
        log_info "Passing extra arguments: $extra_pip_args"
    fi

    if [ "$verbose" != "1" ]; then
        log_info "Installation output suppressed; set HF_CLI_VERBOSE_PIP=1 for full logs"
    fi

    # Check if uv is available and use it for faster installation
    if command_exists uv; then
        log_info "Using uv for faster installation"
        local -a uv_flags
        if [ "$verbose" != "1" ]; then
            uv_flags=(--quiet)
        else
            uv_flags=()
        fi

        # shellcheck disable=SC2086
        run_command "Failed to install $package_spec" uv pip install --python "$VENV_DIR/bin/python" --upgrade ${uv_flags[*]} "$package_spec" $extra_pip_args
    else
        local -a pip_flags
        if [ "$verbose" != "1" ]; then
            pip_flags=(--quiet --progress-bar off --disable-pip-version-check)
        else
            pip_flags=()
        fi

        # shellcheck disable=SC2086
        run_command "Failed to install $package_spec" "$VENV_DIR/bin/python" -m pip install --upgrade "$package_spec" ${pip_flags[*]} $extra_pip_args
    fi
}

# Install huggingface_hub with CLI extras
install_hf_hub() {
    log_info "Installing/upgrading Hugging Face CLI (latest)..."

    install_package "huggingface_hub"
}

# Check if transformers is installed in the venv (using importlib.metadata for speed)
transformers_installed() {
    "$VENV_DIR/bin/python" -c "import importlib.metadata; importlib.metadata.version('transformers')" >/dev/null 2>&1
}

# Install transformers CLI
install_transformers() {
    # Install if --with-transformers was passed OR if transformers is already in the venv
    if [ "$WITH_TRANSFORMERS" != "true" ] && ! transformers_installed; then
        return
    fi

    log_info "Installing/upgrading transformers CLI (latest)..."
    install_package "transformers"
}

# Expose the hf CLI by linking or copying the console script from the virtualenv
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

# Update PATH if needed
update_path() {
    local shell_rc=""
    local -a shell_rc_candidates=()

    # Broaden shell detection and guidance for PATH propagation
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

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    if [ -x "$BIN_DIR/hf" ]; then
        # Test the CLI
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

# Uninstall function 
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

# Main installation process
main() {
    log_info "Installing Hugging Face CLI..."
    log_info "OS: $(detect_os)"
    log_info "Force reinstall: $FORCE_REINSTALL"
    log_info "Install dir: $HF_CLI_DIR"
    log_info "Bin dir: $BIN_DIR"
    log_info "Skip PATH update: $SKIP_PATH_UPDATE"

    ensure_python
    create_directories
    create_venv
    install_hf_hub
    install_transformers
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
    log_info "Try it now: env PATH=\"$BIN_DIR:\$PATH\" hf --help"
    log_info "Examples:"
    log_info "  hf auth login"
    log_info "  hf download deepseek-ai/DeepSeek-R1"
    log_info "  hf jobs run python:3.12 python -c 'print(\"Hello from HF CLI!\")'"
    log_info ""
}

# Handle Ctrl+C
trap 'log_error "Installation interrupted"; exit 130' INT

main "$@"
