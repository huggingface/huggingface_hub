# Hugging Face CLI Installer for Windows
# Usage: powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
# Or: curl -LsSf https://hf.co/cli/install.ps1 | pwsh -

<#
.SYNOPSIS
Installs the Hugging Face CLI on Windows by creating an isolated virtual environment and exposing the `hf` command.

.DESCRIPTION
Downloads and installs the `huggingface_hub` package into a dedicated virtual environment, then copies the generated `hf.exe` console script to a directory on the user's PATH.

.PARAMETER Force
Recreates the virtual environment even if it already exists. Off by default.

.PARAMETER Verbose
Enables verbose output, including detailed pip logs.

.PARAMETER NoModifyPath
Skips PATH modifications; `hf` must be invoked via its full path unless you add it manually.

.PARAMETER WithTransformers
Also install the transformers CLI.

.EXAMPLE
powershell -c "irm https://hf.co/cli/install.ps1 | iex"
powershell -c "irm https://hf.co/cli/install.ps1 | iex" -WithTransformers
#>

<#
.NOTES
Environment variables:
  HF_HOME           Installation base directory; installer uses $env:HF_HOME\cli when set
  HF_CLI_BIN_DIR    Directory for the hf wrapper (default: $env:USERPROFILE\.local\bin)
#>

param(
    [switch]$Force = $false,
    [switch]$Verbose,
    [switch]$NoModifyPath,
    [switch]$WithTransformers = $false
)

$script:LogLevel = if ($Verbose) { 2 } else { 1 }
$script:PathUpdated = $false

if ($Verbose) {
    $env:HF_CLI_VERBOSE_PIP = '1'
}

# Set error action preference
$ErrorActionPreference = "Stop"

# Colors for output
$Colors = @{
    Red = 'Red'
    Green = 'Green'
    Yellow = 'Yellow'
    Blue = 'Blue'
    White = 'White'
}

function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )

    $timestamp = Get-Date -Format "HH:mm:ss"
    switch ($Level) {
        "DEBUG" {
            if ($script:LogLevel -lt 2) { return }
            Write-Host "[$timestamp] [DEBUG] $Message" -ForegroundColor $Colors.Blue
        }
        "INFO" {
            if ($script:LogLevel -lt 1) { return }
            Write-Host "[$timestamp] [INFO] $Message" -ForegroundColor $Colors.Blue
        }
        "SUCCESS" {
            Write-Host "[$timestamp] [SUCCESS] $Message" -ForegroundColor $Colors.Green
        }
        "WARNING" {
            Write-Host "[$timestamp] [WARNING] $Message" -ForegroundColor $Colors.Yellow
        }
        "ERROR" {
            Write-Host "[$timestamp] [ERROR] $Message" -ForegroundColor $Colors.Red
        }
    }
}

# Normalize user-supplied paths
function Resolve-CliPath {
    param([string]$Path)

    if (-not $Path) { return $null }

    $expanded = [Environment]::ExpandEnvironmentVariables($Path)

    if ($expanded -eq '~') { return $env:USERPROFILE }

    if ($expanded -like '~\*') {
        $suffix = $expanded.Substring(2)
        if ([string]::IsNullOrWhiteSpace($suffix)) { return $env:USERPROFILE }
        return (Join-Path $env:USERPROFILE $suffix)
    }

    if ([System.IO.Path]::IsPathRooted($expanded)) {
        return [System.IO.Path]::GetFullPath($expanded)
    }

    $base = (Get-Location).ProviderPath
    return [System.IO.Path]::GetFullPath((Join-Path $base $expanded))
}

# Compose installer directories using environment variables only
if ($env:HF_HOME) {
    $HF_CLI_DIR = Resolve-CliPath (Join-Path $env:HF_HOME "cli")
} else {
    $HF_CLI_DIR = Resolve-CliPath (Join-Path $env:USERPROFILE ".hf-cli")
}

$VENV_DIR = Join-Path $HF_CLI_DIR "venv"

if ($env:HF_CLI_BIN_DIR) {
    $BIN_DIR = Resolve-CliPath $env:HF_CLI_BIN_DIR
} else {
    $BIN_DIR = Resolve-CliPath (Join-Path $env:USERPROFILE ".local\bin")
}

$SCRIPTS_DIR = Join-Path $VENV_DIR "Scripts"
$script:VenvPython = $null

function Test-Command {
    param([string]$Command)
    try { Get-Command $Command -ErrorAction Stop | Out-Null; return $true } catch { return $false }
}

function Test-PythonVersion {
    param([string]$PythonCmd)
    try {
        $version = & $PythonCmd --version 2>&1
        if ($version -match "Python 3\.(\d+)\.") {
            $minorVersion = [int]$matches[1]
            return $minorVersion -ge 9 # Python 3.9+
        }
        return $false
    } catch { return $false }
}

function Find-Python {
    Write-Log "Looking for Python 3.9+ installation..."

    # Try common Python commands
    $pythonCommands = @("python", "python3", "py")

    foreach ($cmd in $pythonCommands) {
        if (Test-Command $cmd) {
            if (Test-PythonVersion $cmd) {
                $version = & $cmd --version 2>&1
                Write-Log "Found compatible Python: $version using command '$cmd'"
                return $cmd
            }
        }
    }

    # Try Python Launcher for Windows
    if (Test-Command "py") {
        try {
            $version = py -3 --version 2>&1
            if ($version -match "Python 3\.(\d+)\.") {
                $minorVersion = [int]$matches[1]
                if ($minorVersion -ge 9) {
                    Write-Log "Found compatible Python: $version using Python Launcher"
                    return "py -3"
                }
            }
        } catch { }
    }

    Write-Log "Python 3.9+ is required but not found." "ERROR"
    Write-Log "Please install Python from https://python.org or Microsoft Store" "ERROR"
    Write-Log "Make sure to check 'Add Python to PATH' during installation" "ERROR"
    throw "Python 3.9+ not found"
}

function New-Directories {
    Write-Log "Creating directories..."
    if (-not (Test-Path $HF_CLI_DIR)) { New-Item -ItemType Directory -Path $HF_CLI_DIR -Force | Out-Null }
    if (-not (Test-Path $BIN_DIR)) { New-Item -ItemType Directory -Path $BIN_DIR -Force | Out-Null }
}

function New-VirtualEnvironment {
    param([string]$PythonCmd)

    Write-Log "Creating virtual environment..."

    if (Test-Path $VENV_DIR) {
        if ($Force) {
            Write-Log "Virtual environment already exists; removing it since --force was passed" "WARNING"
            Remove-Item -Path $VENV_DIR -Recurse -Force
        } else {
            Write-Log "Virtual environment already exists. Use -Force to recreate." "WARNING"
            Write-Log "Skipping virtual environment creation..."
            return
        }
    }

    # Fail fast when venv module is unavailable
    try {
        if ($PythonCmd -eq "py -3") {
            & py -3 -c "import venv" | Out-Null
        } else {
            & $PythonCmd -c "import venv" | Out-Null
        }
    } catch {
        Write-Log "Python installation is missing the venv module." "ERROR"
        Write-Log "Install the optional venv feature or repair Python before retrying." "ERROR"
        Write-Log "Microsoft Store Python: Repair via Apps settings" "INFO"
        Write-Log "python.org installer: Choose 'Modify' and enable 'pip/venv'." "INFO"
        throw "Python venv module unavailable"
    }

    # Create virtual environment
    if ($PythonCmd -eq "py -3") {
        & py -3 -m venv $VENV_DIR
    } else {
        & $PythonCmd -m venv $VENV_DIR
    }
    if (-not $?) { throw "Failed to create virtual environment" }

    # Mark this installation as installer-managed
    $markerFile = Join-Path $VENV_DIR ".hf_installer_marker"
    New-Item -Path $markerFile -ItemType File -Force | Out-Null

    # Use the venv's python -m pip for deterministic upgrades
    $script:VenvPython = Join-Path $SCRIPTS_DIR "python.exe"
    Write-Log "Upgrading pip..."
    & $script:VenvPython -m pip install --upgrade pip
    if (-not $?) { throw "Failed to upgrade pip" }
}

function Install-Package {
    param([string]$PackageSpec)

    if (-not $script:VenvPython) { $script:VenvPython = Join-Path $SCRIPTS_DIR "python.exe" }

    $extraArgsRaw = if ($env:HF_CLI_PIP_ARGS) { $env:HF_CLI_PIP_ARGS } else { $env:HF_PIP_ARGS }

    if ($extraArgsRaw) {
        Write-Log "Passing extra arguments: $extraArgsRaw"
    }

    if ($env:HF_CLI_VERBOSE_PIP -ne '1') {
        Write-Log "Installation output suppressed; set HF_CLI_VERBOSE_PIP=1 for full logs"
    }

    # Check if uv is available and use it for faster installation
    if (Test-Command "uv") {
        Write-Log "Using uv for faster installation"
        $uvArgs = @('pip', 'install', '--python', $script:VenvPython, '--upgrade')

        if ($env:HF_CLI_VERBOSE_PIP -ne '1') {
            $uvArgs += '--quiet'
        }

        $uvArgs += $PackageSpec

        if ($extraArgsRaw) {
            $uvArgs += $extraArgsRaw -split '\s+'
        }

        & uv @uvArgs
        if (-not $?) { throw "Failed to install $PackageSpec with uv" }
    }
    else {
        $pipArgs = @('-m', 'pip', 'install', '--upgrade')

        if ($env:HF_CLI_VERBOSE_PIP -ne '1') {
            $pipArgs += @('--quiet', '--progress-bar', 'off', '--disable-pip-version-check')
        }

        $pipArgs += $PackageSpec

        if ($extraArgsRaw) {
            $pipArgs += $extraArgsRaw -split '\s+'
        }

        & $script:VenvPython @pipArgs
        if (-not $?) { throw "Failed to install $PackageSpec" }
    }
}

function Install-HuggingFaceHub {
    Write-Log "Installing/upgrading Hugging Face CLI (latest)..."

    Install-Package -PackageSpec 'huggingface_hub'
}

function Test-TransformersInstalled {
    # Use importlib.metadata for speed (avoids loading the full transformers module)
    if (-not $script:VenvPython) { $script:VenvPython = Join-Path $SCRIPTS_DIR "python.exe" }
    try {
        & $script:VenvPython -c "import importlib.metadata; importlib.metadata.version('transformers')" 2>&1 | Out-Null
        return $?
    } catch {
        return $false
    }
}

function Install-Transformers {
    # Install if -WithTransformers was passed OR if transformers is already in the venv
    if (-not $WithTransformers -and -not (Test-TransformersInstalled)) {
        return
    }

    Write-Log "Installing/upgrading transformers CLI (latest)..."
    Install-Package -PackageSpec 'transformers'
}

function Publish-HfCommand {
    Write-Log "Copying hf CLI launcher..."

    $hfExeSource = Join-Path $SCRIPTS_DIR "hf.exe"
    if (-not (Test-Path $hfExeSource)) {
        throw "hf.exe not found in virtual environment. Check that The Hugging Face CLI installed correctly."
    }

    $hfExeTarget = Join-Path $BIN_DIR "hf.exe"
    Copy-Item -Path $hfExeSource -Destination $hfExeTarget -Force

    $hfScriptSource = Join-Path $SCRIPTS_DIR "hf-script.py"
    if (Test-Path $hfScriptSource) {
        Copy-Item -Path $hfScriptSource -Destination (Join-Path $BIN_DIR "hf-script.py") -Force
    }

    Write-Log "hf CLI available at $hfExeTarget"
    Write-Log ('Run without updating PATH: & "{0}" --help' -f $hfExeTarget)
}

function Update-Path {
    Write-Log "Checking PATH configuration..."

    # Get current user PATH
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")

    if ($currentPath -notlike "*$BIN_DIR*") {
        Write-Log "Adding $BIN_DIR to user PATH..."

        try {
            $newPath = "$BIN_DIR;" + $currentPath
            [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")

            # Update PATH for current session
            $env:PATH = "$BIN_DIR;$env:PATH"

            Write-Log "Added $BIN_DIR to PATH. Changes will take effect in new terminals." "SUCCESS"
            Write-Log "Current PowerShell session already includes hf after this update." "INFO"
            Write-Log "Undo later via Settings ▸ Environment Variables, or: [Environment]::SetEnvironmentVariable(`"PATH`", ($([Environment]::GetEnvironmentVariable('PATH','User')) -replace [regex]::Escape(`"$BIN_DIR;`"), ''), 'User')" "INFO"
            $script:PathUpdated = $true
        }
        catch {
            Write-Log "Failed to update PATH automatically. Please add manually:" "WARNING"
            Write-Log @"
Run: [Environment]::SetEnvironmentVariable("PATH", "$BIN_DIR;$([Environment]::GetEnvironmentVariable('PATH','User'))", 'User')
"@ "WARNING"
        }
    } else {
        Write-Log "PATH already contains $BIN_DIR"
    }
}

function Test-Installation {
    Write-Log "Verifying installation..."

    $hfExecutable = Join-Path $BIN_DIR "hf.exe"

    if (Test-Path $hfExecutable) {
        try {
            # Test the CLI
            $output = & $hfExecutable version 2>&1
            if ($?) {
                Write-Log "CLI location: $hfExecutable"
                Write-Log "Installation directory: $HF_CLI_DIR"
                return $true
            } else {
                Write-Log "Installation verification failed. The hf command is not working properly." "ERROR"
                Write-Log "Error output: $output" "ERROR"
                return $false
            }
        }
        catch {
            Write-Log "Installation verification failed: $($_.Exception.Message)" "ERROR"
            return $false
        }
    } else {
        Write-Log "Installation failed. hf.exe not found in $BIN_DIR." "ERROR"
        return $false
    }
}

function Show-UninstallInfo {
    Write-Log ""
    Write-Log "To uninstall the Hugging Face CLI:"
    Write-Log "  Remove-Item -Path '$HF_CLI_DIR' -Recurse -Force"
    Write-Log "  Remove-Item -Path '$BIN_DIR\hf.exe'"
    Write-Log "  Remove-Item -Path '$BIN_DIR\hf-script.py' (if present)"
    Write-Log ""
    if ($script:PathUpdated) {
        Write-Log "Remove '$BIN_DIR' from your user PATH via Settings ▸ Environment Variables," "INFO"
        Write-Log "or run: [Environment]::SetEnvironmentVariable(`"PATH`", ($([Environment]::GetEnvironmentVariable('PATH','User')) -replace [regex]::Escape(`"$BIN_DIR;`"), ''), 'User')" "INFO"
    } elseif ($NoModifyPath) {
        Write-Log 'PATH was not modified (--no-modify-path).' 'INFO'
    } else {
        Write-Log "If you added '$BIN_DIR' to PATH manually, remove it when finished." "INFO"
    }
}

function Show-Usage {
    Write-Log ''
    Write-Log 'Usage examples:'
    Write-Log '  hf auth login'
    Write-Log '  hf download deepseek-ai/DeepSeek-R1'
    Write-Log '  hf jobs run python:3.12 python -c ''print("Hello from HF CLI!")'''
    Write-Log ''
    Write-Log "The 'hf' command is now available." 'SUCCESS'
    Write-Log 'Please **close and reopen** your terminal to use it directly, for example: `hf --help`' 'INFO'
    Write-Log 'Alternatively, you can test it immediately in this session by using the full path:' 'INFO'

    $hfExecutable = Join-Path $BIN_DIR 'hf.exe'
    Write-Log ('  & "{0}" --help' -f $hfExecutable)
}

# Main installation process
function Main {
    try {
        Write-Log "Installing Hugging Face CLI for Windows..."
        Write-Log "PowerShell version: $($PSVersionTable.PSVersion)"

        $pythonCmd = Find-Python
        New-Directories
        New-VirtualEnvironment -PythonCmd $pythonCmd
        Install-HuggingFaceHub
        Install-Transformers
        Publish-HfCommand
        if ($NoModifyPath) {
            Write-Log 'Skipping PATH modification (--no-modify-path).'
        } else {
            Update-Path
        }

        if (Test-Installation) {
            $hfExecutable = Join-Path $BIN_DIR "hf.exe"
            Show-Usage
            Show-UninstallInfo
            Write-Log "hf CLI ready!" "SUCCESS"
            Write-Log "Binary: $hfExecutable"
            Write-Log "Virtualenv: $HF_CLI_DIR"
        } else {
            throw "Installation verification failed"
        }
    }
    catch {
        Write-Log "Installation failed: $($_.Exception.Message)" "ERROR"
        exit 1
    }
}

# Handle Ctrl+C
$null = Register-ObjectEvent -InputObject ([Console]) -EventName CancelKeyPress -Action {
    Write-Log "Installation interrupted" "ERROR"
    exit 130
}

# Run main function
Main
