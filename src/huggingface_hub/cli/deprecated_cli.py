"""Deprecated `huggingface-cli` entry point. Warns and exits."""

import sys

from ._output import out


def main() -> None:
    out.warning(
        "\n"
        "  ⚠️  `huggingface-cli` is deprecated and no longer works.\n"
        "\n"
        "  Use `hf` instead:\n"
        "    curl -LsSf https://hf.co/cli/install.sh | bash\n"
        "    brew install hf\n"
        "    pip install hf\n"
        "\n"
        "  Examples:\n"
        "    hf auth login\n"
        "    hf download unsloth/gemma-4-31B-it-GGUF\n"
        "    hf upload my-cool-model . .\n"
        '    hf models ls --search "gemma"\n'
        "    hf repos ls --format json\n"
        "    hf jobs run python:3.12 python -c 'print(\"Hello!\")'\n"
        "    hf --help\n",
    )
    sys.exit(1)
