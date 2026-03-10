<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Build a CLI extension

The `hf` CLI supports extensions — third-party commands that anyone can build, publish, and install. Once installed, an extension named `hf-<name>` becomes available as `hf <name>`, just like a built-in command. Extensions are hosted on public GitHub repositories and installed with `hf extensions install`.

> [!TIP]
> For the full reference of extension management commands (`install`, `list`, `remove`, `exec`), see the [CLI reference](../package_reference/cli.md).

## Naming convention

Your GitHub repository **must** be named `hf-<name>`, where `<name>`:

- Contains only letters, digits, `.`, `_`, and `-`
- Starts with a letter or digit
- Does not shadow a built-in command (`auth`, `download`, `upload`, `cache`, `repo`, etc.)

```
✅  hf-claude
✅  hf-my-tool
✅  hf-mem
❌  hf--bad (starts with -)
❌  hf-download (shadows built-in)
```

When users install, they can omit the owner to default to the `huggingface` org:

```bash
hf extensions install hf-claude            # installs huggingface/hf-claude
hf extensions install alice/hf-my-tool     # installs alice/hf-my-tool
```

## Binary extensions

The simplest type: a single executable file named `hf-<name>` placed at the root of your repository. This can be a shell script, a compiled Go/Rust binary, or anything else that's executable.

### Example: shell script

Create a repository `hf-greet` with a single file `hf-greet` at the root:

```bash
#!/usr/bin/env bash
echo "Hello from hf-greet! Args: $@"
```

Make it executable:

```bash
chmod +x hf-greet
```

Push to GitHub and users can install it:

```bash
hf extensions install your-username/hf-greet
hf greet world
# Hello from hf-greet! Args: world
```

> [!NOTE]
> On POSIX systems, if the binary cannot be executed directly (e.g. missing shebang), the CLI falls back to running it with `sh`.

### Compiled binaries

For Go, Rust, or other compiled languages, place the compiled binary named `hf-<name>` at the repository root. Make sure to compile for the target platform(s) your users need.

## Python extensions

For extensions that need Python dependencies, you can ship a pip-installable package instead of a standalone binary.

### How it works

1. The CLI checks for a binary named `hf-<name>` at the repository root
2. If no binary is found, it installs the repository as a Python package in an isolated virtual environment at `~/.local/share/hf/extensions/hf-<name>/venv/`
3. The package must expose a console script entry point named `hf-<name>`

### Project structure

```
hf-mem/
├── pyproject.toml
└── src/
    └── hf_mem/
        └── __init__.py
```

### pyproject.toml

The key requirement is the `[project.scripts]` entry point:

```toml
[project]
name = "hf-mem"
version = "0.1.0"
description = "Show system memory info"
requires-python = ">=3.9"
dependencies = ["psutil"]

[project.scripts]
hf-mem = "hf_mem:main"

[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.backends._legacy:_Backend"
```

The console script name **must** match `hf-<name>` — this is how the CLI locates the executable after installation.

### Entry point

```python
# src/hf_mem/__init__.py
import psutil

def main():
    mem = psutil.virtual_memory()
    print(f"Total: {mem.total / 1e9:.1f} GB")
    print(f"Used:  {mem.used / 1e9:.1f} GB ({mem.percent}%)")
```

## Provide a description

Descriptions are shown in `hf --help` and `hf extensions list`. You can set one in two ways:

### Option 1: `manifest.json`

Add a `manifest.json` at the repository root:

```json
{
    "description": "Chat with Claude from your terminal"
}
```

### Option 2: `pyproject.toml`

For Python extensions, the `description` field in `pyproject.toml` is used as a fallback:

```toml
[project]
description = "Show system memory info"
```

The CLI checks `manifest.json` first, then `pyproject.toml`, then the GitHub repository description.

## Test locally

You can test your extension without publishing by placing it directly in the extensions directory.

### Binary extension

```bash
mkdir -p ~/.local/share/hf/extensions/hf-greet
cp hf-greet ~/.local/share/hf/extensions/hf-greet/
```

Create a `manifest.json` in the same directory for proper registration:

```bash
cat > ~/.local/share/hf/extensions/hf-greet/manifest.json << 'EOF'
{
    "owner": "local",
    "repo": "hf-greet",
    "repo_id": "local/hf-greet",
    "short_name": "greet",
    "executable_name": "hf-greet",
    "executable_path": "~/.local/share/hf/extensions/hf-greet/hf-greet",
    "type": "binary",
    "installed_at": "2026-01-01T00:00:00+00:00",
    "source": "local"
}
EOF
```

Then run:

```bash
hf greet --help
```

### Python extension

For Python extensions, create the venv manually:

```bash
EXT_DIR=~/.local/share/hf/extensions/hf-mem
mkdir -p "$EXT_DIR"
python -m venv "$EXT_DIR/venv"
"$EXT_DIR/venv/bin/pip" install /path/to/your/hf-mem
```

Then create a `manifest.json` with `"type": "python"` and `"executable_path"` pointing to the venv binary (e.g. `~/.local/share/hf/extensions/hf-mem/venv/bin/hf-mem`).

## Publish

To publish your extension:

1. Create a **public** GitHub repository named `hf-<name>`
2. For binary extensions: place the executable `hf-<name>` at the repository root
3. For Python extensions: make the repository pip-installable with a `hf-<name>` console script
4. Optionally add a `manifest.json` with a description

Users install with:

```bash
hf extensions install your-username/hf-<name>
```

## How users run your extension

Once installed, there are two ways to invoke an extension:

```bash
# Top-level shorthand (recommended)
hf <name> [ARGS]

# Explicit form
hf extensions exec <name> -- [ARGS]
```

All arguments are passed through to the extension unchanged. For example:

```bash
hf claude --model zai-org/GLM-5
hf extensions exec claude -- --help
```

## Managing extensions

Users can list and remove installed extensions:

```bash
# List all installed extensions
hf extensions list

# Remove an extension
hf extensions remove <name>
```

Aliases `ext` for `extensions`, `ls` for `list`, and `rm` for `remove` are also supported.
