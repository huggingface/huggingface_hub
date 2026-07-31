<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub.svg">
    <img alt="huggingface_hub library logo" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p> 

<p align="center">
    <i>The official Python client for the Huggingface Hub.</i>
</p>

<p align="center">
    <a href="https://huggingface.co/docs/huggingface_hub/en/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/huggingface_hub/index.svg?down_color=red&down_message=offline&up_message=online&label=doc"></a>
    <a href="https://github.com/huggingface/huggingface_hub/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/huggingface_hub.svg"></a>
    <a href="https://github.com/huggingface/huggingface_hub"><img alt="PyPi version" src="https://img.shields.io/pypi/pyversions/huggingface_hub.svg"></a>
    <a href="https://pypi.org/project/huggingface-hub"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/huggingface_hub"></a>
    <a href="https://codecov.io/gh/huggingface/huggingface_hub"><img alt="Code coverage" src="https://codecov.io/gh/huggingface/huggingface_hub/branch/main/graph/badge.svg?token=RXP95LE2XL"></a>
</p>

---

**CLI Documentation**: <a href="https://huggingface.co/docs/huggingface_hub/guides/cli" target="_blank">https://huggingface.co/docs/huggingface_hub/guides/cli</a>

**Source Code**: <a href="https://github.com/huggingface/huggingface_hub" target="_blank">https://github.com/huggingface/huggingface_hub</a>

> [!TIP]
> This package provides a clean CLI interface via `uvx hf`. It is **not meant to be used as a package in scripts**. Use `huggingface_hub` instead.

## Usage

Install and use the CLI with `uv`:

```bash
uvx hf version
uvx hf auth whoami
uvx hf download MiniMaxAI/MiniMax-M2
```

## Note

The legacy `hf` package (which provided a Mapping interface to HuggingFace) has been moved to [hfdol](https://pypi.org/project/hfdol/).