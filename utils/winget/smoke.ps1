# Test the exact ZIP in a relocated directory and through a real WinGet install.
$ErrorActionPreference = 'Stop'
$env:HF_HUB_DISABLE_UPDATE_CHECK = '1'
$env:HF_ENDPOINT = 'https://huggingface.co'
$testRoot = Join-Path $env:RUNNER_TEMP 'hf winget smoke'
New-Item -ItemType Directory -Force $testRoot | Out-Null
$env:HF_HOME = Join-Path $testRoot 'cache'
$originalUserProfile = $env:USERPROFILE
$env:USERPROFILE = $testRoot
$portable = Join-Path $testRoot 'portable with spaces'
Expand-Archive dist/hf-windows-x86_64.zip $portable

function Invoke-Hf {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "hf failed: $Arguments" }
}

$exe = Join-Path $portable 'hf.exe'
Invoke-Hf $exe @('--help')
Invoke-Hf $exe @('version')
Invoke-Hf $exe @('update')
Invoke-Hf $exe @('skills', 'add', 'hf-cli', '--global')
Invoke-Hf $exe @('download', 'hf-internal-testing/tiny-random-bert', 'pytorch_model.bin')
# Exercise the bundled native extension, and ensure a Python extension can bootstrap pip.
& "$portable/python/python.exe" -I -c "import hf_xet, venv; venv.create(r'$testRoot/extension env', with_pip=True)"
if ($LASTEXITCODE -ne 0) { throw 'Bundled Python/Xet/venv smoke test failed' }

$env:USERPROFILE = $originalUserProfile

# WinGet is not available by default in every hosted runner image.
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Install-Module Microsoft.WinGet.Client -Force -Scope CurrentUser -Repository PSGallery
    Repair-WinGetPackageManager -AllUsers -Force
}
# Serve the exact release ZIP locally; production manifests retain their versioned GitHub URL.
$server = Start-Process python -ArgumentList @('-m', 'http.server', '8765', '--bind', '127.0.0.1', '--directory', 'dist') -PassThru
$manifests = Join-Path $testRoot 'manifests'
Copy-Item dist/manifests $manifests -Recurse
$installer = Get-ChildItem $manifests -Recurse -Filter '*.installer.yaml' | Select-Object -First 1
(Get-Content $installer.FullName -Raw) -replace 'https://github.com/huggingface/huggingface_hub/releases/download/[^/]+/hf-windows-x86_64.zip', 'http://127.0.0.1:8765/hf-windows-x86_64.zip' | Set-Content $installer.FullName
$manifestDir = $installer.DirectoryName
# Register the same ZIP as an older version to exercise WinGet's upgrade path.
$oldManifestDir = Join-Path $testRoot 'old manifests'
New-Item -ItemType Directory $oldManifestDir | Out-Null
Get-ChildItem $manifestDir -Filter '*.yaml' | ForEach-Object {
    (Get-Content $_.FullName -Raw) -replace '(?m)^PackageVersion: .*', 'PackageVersion: 0.0.1' |
        Set-Content (Join-Path $oldManifestDir $_.Name)
}
try {
    # Wait until the local asset server is ready.
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        try { Invoke-WebRequest http://127.0.0.1:8765/ -UseBasicParsing | Out-Null; break }
        catch { if ($attempt -eq 19) { throw }; Start-Sleep -Milliseconds 250 }
    }
    winget settings --enable LocalManifestFiles
    if ($LASTEXITCODE -ne 0) { throw 'Cannot enable local manifests' }
    winget validate --manifest $manifestDir
    if ($LASTEXITCODE -ne 0) { throw 'Invalid WinGet manifests' }
    winget install --manifest $oldManifestDir --accept-package-agreements --accept-source-agreements --disable-interactivity
    if ($LASTEXITCODE -ne 0) { throw 'WinGet install failed' }
    $installed = Join-Path $env:LOCALAPPDATA 'Microsoft/WinGet/Links/hf.exe'
    Invoke-Hf $installed @('version')
    Invoke-Hf $installed @('update')
    winget upgrade --manifest $manifestDir --accept-package-agreements --accept-source-agreements --disable-interactivity
    if ($LASTEXITCODE -ne 0) { throw 'WinGet upgrade failed' }
    Invoke-Hf $installed @('--help')
} finally {
    winget uninstall --id HuggingFace.hf --exact --disable-interactivity
    $uninstallExit = $LASTEXITCODE
    Stop-Process -Id $server.Id
    if ($uninstallExit -ne 0) { throw 'WinGet uninstall failed' }
}
if (Test-Path $installed) { throw 'WinGet launcher remains after uninstall' }
if (-not (Test-Path $env:HF_HOME)) { throw 'Uninstall removed the user cache' }
