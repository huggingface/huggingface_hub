# Windows CLI / WinGet

The Windows x86_64 (Intel/AMD 64-bit) distribution bundles a full Python runtime,
`huggingface_hub`, and its dependencies, including `hf-xet`. No system Python is
needed. The small native launcher resolves WinGet's symlink and launches the
bundled interpreter in isolated mode, keeping Python subprocesses and extension
venvs usable. This avoids the compatibility changes needed for a frozen application.

Like [Git-Xet in xet-core](https://github.com/huggingface/xet-core/blob/main/.github/workflows/git-xet-release.yml),
we publish versioned Windows assets on GitHub releases. Git-Xet also builds a WiX
MSI; here WinGet handles the portable ZIP's registration, PATH and removal, so no
MSI or installer custom actions are needed. ARM64 and 32-bit Windows are outside
this initial scope.

## Build and check

`build-windows-cli.yml` compiles `launcher.c`, runs `build.py`, and tests the ZIP
in a path containing spaces, then installs/upgrades/removes it with WinGet.
It also checks Xet loading, a model download, skills, and venv creation with pip.
The workflow runs on relevant PRs and can be dispatched manually. The runtime
URL and checksum are pinned in `build.py`; update them together for Python updates.

To build locally, use an **x64 Native Tools Command Prompt for Visual Studio**
with Python 3.12 installed, from the repository root:

```bat
mkdir dist
cl /nologo /O2 /MT /W4 /WX utils\winget\launcher.c /Fe:dist\hf.exe /Fo:dist\hf.obj
python -m pip install PyYAML
python utils/winget/build.py
```

Outputs:

- `dist/hf-windows-x86_64.zip`: application, runtime, licenses and dependency versions.
- `dist/hf-windows-x86_64.zip.sha256`: application ZIP checksum.
- `dist/hf-winget-manifests.zip`: the three manifests under their submission path.

Stable releases (including patches) build and attach these assets to the existing
GitHub release. Release candidates do not publish Windows assets. Publishing the
release remains the existing manual step; no WinGet submission happens while it
is a draft. Do not replace published ZIPs: manifests pin their SHA256.

## First submission and subsequent updates

1. Publish a stable GitHub release with the assets above.
2. Download and extract `hf-winget-manifests.zip`. On Windows, validate it:

   ```powershell
   winget validate --manifest manifests/h/HuggingFace/hf/<version>
   ```

3. Submit those three YAML files to `microsoft/winget-pkgs` and address Microsoft's
   validation/review feedback. The proposed package identifier is `HuggingFace.hf`.
4. For automatic submissions after the first accepted package, configure
   `WINGET_GITHUB_TOKEN` as a repository secret (a Winget-Create compatible classic
   GitHub token with `public_repo` access), then set repository variable
   `WINGET_PUBLISH=true`. `publish-winget.yml` submits on stable release publication;
   it can also be dispatched with a public stable tag to retry a failed submission.
   Keep the variable unset until ready to submit. Never run this workflow with PR secrets.

[`wingetcreate submit`](https://github.com/microsoft/winget-create/blob/main/doc/submit.md)
uses the generated manifests for each release, including the first if desired.
Microsoft's acceptance and catalog indexing are separate from publishing on GitHub.
The default Actions token cannot submit to the external manifests repository.

Users update with `winget upgrade --id HuggingFace.hf --exact --source winget`.
`hf update` prints this command and exits before any update check or subprocess:
Windows locks the running launcher and runtime. WinGet handles available-version
checks, avoiding notices about PyPI releases that are not in its catalog yet.
`hf skills add -g` remains explicit; installing the ZIP does not run the PowerShell
installer's automatic skill installation. User tokens, model caches and installed
skills remain outside the application directory and survive uninstalling it.
