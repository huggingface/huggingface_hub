<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Create a CLI extension

The `hf` CLI supports extensions, custom commands provided by the community that integrate seamlessly
into the CLI. Extensions are hosted as public GitHub repositories and can be installed with a single command.
Once installed, they appear as top-level `hf` commands just like built-in ones.

This system is inspired by [GitHub CLI extensions](https://docs.github.com/en/github-cli/github-cli/creating-github-cli-extensions).
In this guide, you will learn how to create your own extension, publish it, and make it discoverable.

> [!TIP]
> For user-facing documentation on installing and managing extensions, see the
> [CLI reference for `hf extensions`](../package_reference/cli.md#hf-extensions).

## Overview

There are two types of extensions:

1. **Shell script extensions**: a single executable file (bash script, compiled binary, etc.) placed at the
   root of the repository.
2. **Python extensions**: a standard Python package with a `pyproject.toml`. Installed in an isolated virtual
   environment so dependencies don't conflict with the user's system.

Both types share the same conventions:

- The GitHub repository **must** be named `hf-<name>` (e.g., `hf-claude`, `hf-mem`).
- Once installed, users run the extension with `hf <name>` (e.g., `hf claude`).
- The extension is listed in `hf --help` under "Extension commands".

When a user runs `hf extensions install [OWNER/]hf-<name>`, the system first looks for a binary/script file
named `hf-<name>` at the repository root. If found, it installs as a shell script extension. Otherwise, it
falls back to installing the repo as a Python package.

## Create a shell script extension

A shell script extension is the simplest type. You only need a GitHub repository with an executable file
named `hf-<name>` at the root.

### Minimal example

Create a repository named `hf-hello` on GitHub with a single file:

**`hf-hello`** (at the repository root):

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "Hello from hf-hello extension!"
echo "Arguments: $@"
```

That's it! Users can now install and run it:

```bash
>>> hf extensions install <your-username>/hf-hello
>>> hf hello
Hello from hf-hello extension!
Arguments:
```

### Tips for shell script extensions

- Always start with a shebang (`#!/usr/bin/env bash`) and `set -euo pipefail` for safety.
- The script receives all extra arguments passed by the user. For example, `hf hello --name world`
  passes `--name world` to the script.
- You can access the user's Hugging Face token via the `HF_TOKEN` environment variable if they are logged in.
- Add a `manifest.json` at the repository root to provide a description (see [Add a description](#add-a-description)).
- External dependencies (e.g., `fzf`, `jq`, etc.) are **not** installed automatically with your extension. Check
  for required tools at the start of your script and fail gracefully with a helpful error message if they are missing.

> [!TIP]
> For a real-world example, see [hanouticelina/hf-claude](https://github.com/hanouticelina/hf-claude) —
> a shell script extension that launches Claude Code with HF Inference Providers.

## Create a Python extension

Python extensions are full Python packages installed in an isolated virtual environment. This is the best
choice when your extension has Python dependencies or more complex logic.

### Minimal example

Create a repository named `hf-hello` on GitHub with this structure:

```
hf-hello/
├── pyproject.toml
└── src/
    └── hf_hello/
        ├── __init__.py
        └── cli.py
```

**`pyproject.toml`**:

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "hf-hello"
version = "0.1.0"
description = "A hello-world hf CLI extension"
requires-python = ">=3.10"

[project.scripts]
hf-hello = "hf_hello.cli:main"
```

**`src/hf_hello/cli.py`**:

```python
import sys

def main():
    print("Hello from hf-hello extension!")
    print(f"Arguments: {sys.argv[1:]}")

if __name__ == "__main__":
    main()
```

The critical part is the `[project.scripts]` entry point: it **must** be named `hf-<name>` (here, `hf-hello`).
This is how the `hf` CLI discovers and executes your extension.

Users install and run it the same way:

```bash
>>> hf extensions install <your-username>/hf-hello
>>> hf hello
Hello from hf-hello extension!
Arguments: []
```

### How Python extensions are installed

When a user installs a Python extension, the following happens:

1. A virtual environment is created at `~/.local/share/hf/extensions/hf-<name>/venv/`.
2. Your package is installed via `pip install` from the GitHub repo archive.
3. The system verifies that a `hf-<name>` console script was created in the venv.

This means your extension's dependencies are fully isolated — they won't conflict with the user's
other Python packages.

> [!TIP]
> For a real-world example, see [alvarobartt/hf-mem](https://github.com/alvarobartt/hf-mem) —
> a Python extension that estimates inference memory requirements for HF models.

## Add a description

A description helps users understand what your extension does. It appears in `hf extensions list`
and in `hf --help`.

The system looks for a description in the following order:

1. **`manifest.json`** at the repository root:

```json
{
    "description": "A short description of what your extension does"
}
```

2. **`pyproject.toml`** `description` field (for Python extensions):

```toml
[project]
description = "A short description of what your extension does"
```

3. **GitHub repository description** (the "About" field on the repo page).

For Python extensions, setting `description` in `pyproject.toml` is the most natural approach.
For shell script extensions, use a `manifest.json` file or set the GitHub repository description.

## Make your extension discoverable

To help users find your extension, add the **`hf-extension`** topic to your GitHub repository:

1. Go to your repository on GitHub.
2. Click the gear icon next to "About" on the right sidebar.
3. Under "Topics", add `hf-extension`.

This is a community convention that makes it easy to browse all available extensions on the
[hf-extension topic page](https://github.com/topics/hf-extension).

Users can then discover your extension directly from the CLI with `hf extensions search`, which lists
all GitHub repositories tagged with the `hf-extension` topic, sorted by stars:

```bash
>>> hf extensions search
NAME   REPO                    STARS DESCRIPTION                         INSTALLED
------ ----------------------- ----- ----------------------------------- ---------
claude hanouticelina/hf-claude     2 Extension for `hf` CLI to launch... yes
agents hanouticelina/hf-agents       HF extension to run local coding...
```

The `INSTALLED` column shows which extensions are already installed locally. From there, users can
install any listed extension with `hf extensions install <repo>`.

## Test your extension

During development, you can install your extension directly from your GitHub repository:

```bash
# Install from your repo
>>> hf extensions install <your-username>/hf-<name>

# Run it
>>> hf <name>

# Reinstall after making changes (push to GitHub first)
>>> hf extensions install <your-username>/hf-<name> --force

# List installed extensions
>>> hf extensions list

# Remove when done
>>> hf extensions remove <name>
```

> [!TIP]
> Use `--force` to overwrite a previously installed version when testing updates.

## Naming rules

Extension names must follow these rules:

- The GitHub repository must be named `hf-<name>`.
- `<name>` must start with a letter or digit.
- `<name>` can contain letters, digits, `.`, `_`, and `-`.
- `<name>` cannot conflict with a built-in `hf` command (e.g., `download`, `upload`, `auth`).

When installing, users can either specify the full `OWNER/hf-<name>` or just `hf-<name>` (which
defaults to the `huggingface` organization).

## Existing extensions

Here are some community extensions you can use as reference:

| Extension | Type | Description |
|-----------|------|-------------|
| [hanouticelina/hf-claude](https://github.com/hanouticelina/hf-claude) | Shell script | Launch Claude Code with HF Inference Providers |
| [alvarobartt/hf-mem](https://github.com/alvarobartt/hf-mem) | Python | Estimate inference memory requirements for HF models |
