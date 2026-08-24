<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub.svg">
    <img alt="huggingface_hub library logo" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub.svg" width="352" height="59" style="max-width: 100%">
  </picture>
  <br/>
  <br/>
</p> 

<p align="center">
    <i>The official CLI and Python client for the Hugging Face Hub.</i>
    <br/>
    <a href="#what-is-huggingface_hub">About</a>
    ·
    <a href="https://huggingface.co/docs/huggingface_hub">Documentation</a>
    ·
    <a href="https://huggingface.co/docs/huggingface_hub/en/installation">Install</a>
    ·
    <a href="https://huggingface.co/docs/huggingface_hub/en/guides/cli">CLI Guide</a>
    ·
    <a href="https://github.com/huggingface/huggingface_hub/blob/main/CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
    <a href="https://huggingface.co/docs/huggingface_hub/en/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/huggingface_hub/index.svg?down_color=red&down_message=offline&up_message=online&label=doc"></a>
    <a href="https://github.com/huggingface/huggingface_hub/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/huggingface_hub.svg"></a>
    <a href="https://github.com/huggingface/huggingface_hub"><img alt="PyPi version" src="https://img.shields.io/pypi/pyversions/huggingface_hub.svg"></a>
    <a href="https://pypi.org/project/huggingface-hub"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/huggingface_hub"></a>
    <a href="https://codecov.io/gh/huggingface/huggingface_hub"><img alt="Code coverage" src="https://codecov.io/gh/huggingface/huggingface_hub/branch/main/graph/badge.svg?token=RXP95LE2XL"></a>
</p>

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_hi.md">हिंदी</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_cn.md">中文 (简体)</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_kn.md">ಕನ್ನಡ</a>
    </p>
</h4>

## Quick start

Install the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) with the standalone installer:

```bash
# On macOS and Linux.
curl -LsSf https://hf.co/cli/install.sh | bash
```

```powershell
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

Log in, then start working with the Hub:

```bash
# Log in (use --token $HF_TOKEN in non-interactive environments)
hf auth login

# Find models served by Inference Providers
hf models ls --warm

# Download a model
hf download Qwen/Qwen3-0.6B

# Upload files to your own repo
hf upload username/my-cool-model ./model.safetensors

# Sync a local folder to a storage bucket
hf buckets sync ./checkpoints hf://buckets/username/my-bucket

# Run a job on Hugging Face infrastructure
hf jobs run python:3.12 python -c "print('Hello from the cloud!')"

# Discover everything else
hf --help
```

The Hub uses tokens to authenticate applications (see [docs](https://huggingface.co/docs/hub/security-tokens)). Check out the [CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli) for a tour of the main features.

## What is `huggingface_hub`?

The `huggingface_hub` library allows you to interact with the [Hugging Face Hub](https://huggingface.co/), a platform democratizing open-source Machine Learning for creators and collaborators. Discover pre-trained models and datasets for your projects, play with the thousands of machine learning apps hosted on the Hub, or create and share your own models, datasets and demos with the community. Everything ships in one package with two interfaces: the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) for your terminal and the `huggingface_hub` library for Python — both designed to work well for humans and AI agents. Use them to:

- [Download files](https://huggingface.co/docs/huggingface_hub/en/guides/download) from the Hub.
- [Upload files](https://huggingface.co/docs/huggingface_hub/en/guides/upload) to the Hub.
- [Manage your repositories](https://huggingface.co/docs/huggingface_hub/en/guides/repository).
- [Run Inference](https://huggingface.co/docs/huggingface_hub/en/guides/inference) on deployed models.
- [Run Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs) on Hugging Face infrastructure.
- [Search](https://huggingface.co/docs/huggingface_hub/en/guides/search) for models, datasets and Spaces.
- [Share Model Cards](https://huggingface.co/docs/huggingface_hub/en/guides/model-cards) to document your models.
- [Engage with the community](https://huggingface.co/docs/huggingface_hub/en/guides/community) through PRs and comments.
- Do all of the above from the terminal with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

## Built for humans and AI agents

The `hf` CLI is designed for people and coding agents alike: the same commands adapt their output when run by an agent. If you use Claude Code, Codex, Cursor, or another coding agent, install the `hf` CLI Skill — a command reference generated from your installed CLI:

```bash
# for Codex, Cursor, OpenCode, Pi and other agents that load skills from `.agents/skills`
hf skills add
# includes the above + Claude Code
hf skills add --claude
```

Learn more in the [Hugging Face CLI for AI agents guide](https://huggingface.co/docs/hub/agents-cli) and the [announcement blog post](https://huggingface.co/blog/hf-cli-for-agents).

## Use the Python library

Install the `huggingface_hub` package with [pip](https://pypi.org/project/huggingface-hub/) (this also installs the `hf` CLI):

```bash
pip install huggingface_hub
```

We recommend using [`uv`](https://docs.astral.sh/uv/) for a fast and reliable install:

```bash
uv pip install huggingface_hub
```

In order to keep the package minimal by default, `huggingface_hub` comes with optional dependencies useful for some use cases. For example, if you want to use the MCP module, run:

```bash
pip install "huggingface_hub[mcp]"
```

To learn more about installation and optional dependencies, check out the [installation guide](https://huggingface.co/docs/huggingface_hub/en/installation).

### Download files

Download a single file

```py
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="zai-org/GLM-5.2", filename="config.json")
```

Or an entire repository

```py
from huggingface_hub import snapshot_download

snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
```

Files will be downloaded in a local cache folder. More details in [this guide](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache).

### Create a repository

```py
from huggingface_hub import create_repo

create_repo(repo_id="super-cool-model")
```

### Upload files

Upload a single file

```py
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="/home/lysandre/dummy-test/README.md",
    path_in_repo="README.md",
    repo_id="lysandre/test-model",
)
```

Or an entire folder

```py
from huggingface_hub import upload_folder

upload_folder(
    folder_path="/path/to/local/space",
    repo_id="username/my-cool-space",
    repo_type="space",
)
```

More details in the [upload guide](https://huggingface.co/docs/huggingface_hub/en/guides/upload).

## Integrating with the Hub.

We're partnering with cool open source ML libraries to provide free model hosting and versioning. You can find the existing integrations [here](https://huggingface.co/docs/hub/libraries).

The advantages are:

- Free model or dataset hosting for libraries and their users.
- Built-in file versioning, even with very large files, made possible by [Xet](https://huggingface.co/docs/hub/xet/index), the Hub's chunk-deduplicated storage backend.
- In-browser widgets to play with the uploaded models.
- Anyone can upload a new model for your library, they just need to add the corresponding tag for the model to be discoverable.
- Fast downloads! We use Cloudfront (a CDN) to geo-replicate downloads so they're blazing fast from anywhere on the globe.
- Usage stats and more features to come.

If you would like to integrate your library, feel free to open an issue to begin the discussion. We wrote a [step-by-step guide](https://huggingface.co/docs/hub/adding-a-library) with ❤️ showing how to do this integration.

## Contributions (feature requests, bugs, etc.) are super welcome 💙💚💛💜🧡❤️

Everyone is welcome to contribute, and we value everybody's contribution. Code is not the only way to help the community.
Answering questions, helping others, reaching out and improving the documentations are immensely valuable to the community.
We wrote a [contribution guide](https://github.com/huggingface/huggingface_hub/blob/main/CONTRIBUTING.md) to summarize
how to get started to contribute to this repository.
