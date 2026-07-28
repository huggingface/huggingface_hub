# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Read the optional `[tool.hf-jobs]` table from a UV script's PEP 723 header.

Some scripts only run correctly with a specific runtime (a given image, a GPU flavor, a system
interpreter, ...). This lets a script carry that launch configuration with it, so that
`hf jobs uv run script.py` just works instead of silently running with the wrong runtime:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["vllm", "datasets"]
#
# [tool.hf-jobs]
# image   = "vllm/vllm-openai:latest"
# flavor  = "l4x1"
# python  = "/usr/bin/python3"
# env     = { PYTHONPATH = "/usr/local/lib/python3.12/dist-packages" }
# secrets = ["HF_TOKEN"]
# ///
```

`[tool.*]` tables are sanctioned by PEP 723 (it is how `uv` reads `[tool.uv]`) and tools ignore the
tables they don't own, so the block is invisible to a plain `uv run`. Header values are *defaults*:
an explicit CLI flag always wins. See `huggingface_hub/cli/jobs.py` for the merge rules.
"""

import re
import sys
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from huggingface_hub.errors import CLIError
from huggingface_hub.utils import get_session


# Reference regex from PEP 723 (https://peps.python.org/pep-0723/#reference-implementation)
_PEP_723_REGEX = re.compile(r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$")

TABLE_NAME = "tool.hf-jobs"


@dataclass
class UvScriptHeader:
    """The `[tool.hf-jobs]` table of a UV script: the launch configuration the script carries."""

    image: str | None = None
    flavor: str | None = None
    python: str | None = None
    timeout: str | None = None
    name: str | None = None
    namespace: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    secrets: list[str] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    volumes: list[str] = field(default_factory=list)


VALID_KEYS = tuple(f.name for f in fields(UvScriptHeader))


@dataclass
class UvScript:
    """A UV script ready to be submitted, with the launch config it carries (if any)."""

    script: str
    """Path (or command) to pass to the Jobs API. A URL script is downloaded, so this is a local path."""

    header: UvScriptHeader | None = None
    """Parsed `[tool.hf-jobs]` table, or `None` if the script has none."""

    tmp_dir: Any = None
    """Keeps a downloaded script alive on disk: the `TemporaryDirectory` is cleaned up with this object."""


def load_uv_script(script: str) -> UvScript:
    """Resolve a `hf jobs uv run` script argument and read its `[tool.hf-jobs]` header.

    - a local file is read in place
    - an `http(s)` URL is downloaded to a temporary directory and from there shipped to the Job like
      any local script: the URL is fetched exactly once, at submit time, instead of being fetched
      again by the container (the header has to be read client-side anyway, since `image`, `flavor`,
      `secrets`, ... are all decided before the container exists)
    - anything else is a command (e.g. `lighteval`) and carries no header
    """
    # Scripts are read as UTF-8, not with the locale encoding: a downloaded script is written as raw
    # bytes and a header with non-ASCII values must not fail to decode depending on the platform.
    if script.startswith(("http://", "https://")):
        local_path, tmp_dir = _download_script(script)
        header = parse_uv_script_header(Path(local_path).read_text(encoding="utf-8"))
        return UvScript(script=local_path, header=header, tmp_dir=tmp_dir)
    path = Path(script)
    if path.is_file():
        return UvScript(script=script, header=parse_uv_script_header(path.read_text(encoding="utf-8")))
    return UvScript(script=script)


def parse_uv_script_header(text: str) -> UvScriptHeader | None:
    """Parse the `[tool.hf-jobs]` table out of a script's PEP 723 header. Returns `None` if absent."""
    content = _extract_pep_723_block(text)
    # Cheap pre-check: scripts that don't use the feature never need a TOML parser.
    if content is None or "hf-jobs" not in content:
        return None
    table = _load_toml(content).get("tool", {}).get("hf-jobs")
    if table is None:
        return None
    if not isinstance(table, dict):
        raise CLIError(f"'{TABLE_NAME}' must be a table in the script's PEP 723 header (i.e. `[{TABLE_NAME}]`).")

    if unknown := sorted(key for key in table if key not in VALID_KEYS):
        raise CLIError(
            f"Unknown key(s) {', '.join(repr(key) for key in unknown)} in the script's [{TABLE_NAME}] table."
            f" Valid keys are: {', '.join(VALID_KEYS)}."
        )
    return UvScriptHeader(
        image=_as_str(table, "image"),
        flavor=_as_str(table, "flavor"),
        python=_as_str(table, "python"),
        timeout=_as_timeout(table),
        name=_as_str(table, "name"),
        namespace=_as_str(table, "namespace"),
        env=_as_str_table(table, "env"),
        secrets=_as_secret_names(table),
        labels=_as_str_table(table, "labels"),
        volumes=_as_str_list(table, "volumes"),
    )


def _download_script(url: str) -> tuple[str, "tempfile.TemporaryDirectory"]:
    """Download a remote UV script to a temporary directory and return `(local_path, tmp_dir)`."""
    name = Path(urlsplit(url).path).name
    tmp_dir = tempfile.TemporaryDirectory(prefix="hf-jobs-uv-")
    local_path = Path(tmp_dir.name) / (name if name.endswith(".py") else "script.py")
    try:
        response = get_session().get(url)
        response.raise_for_status()
    except Exception as e:
        raise CLIError(f"Could not download the UV script from '{url}': {e}") from e
    local_path.write_bytes(response.content)
    return str(local_path), tmp_dir


def _extract_pep_723_block(text: str) -> str | None:
    """Return the content of the first PEP 723 `script` block, with the `# ` prefixes stripped."""
    # The PEP 723 regex anchors on '\n': normalize line endings first, otherwise a script written on
    # Windows (or fetched over HTTP) silently parses as "no header at all".
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for match in _PEP_723_REGEX.finditer(text):
        if match.group("type") == "script":
            return "".join(
                line[2:] if line.startswith("# ") else line[1:]
                for line in match.group("content").splitlines(keepends=True)
            )
    return None


def _load_toml(content: str) -> dict[str, Any]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # tomllib was added in Python 3.11
        try:
            import tomli as tomllib  # type: ignore[import-not-found,no-redef,unused-ignore]
        except ImportError as e:
            raise CLIError(
                f"Reading the [{TABLE_NAME}] table of a UV script requires a TOML parser."
                " Upgrade to Python 3.11+ or install one with `pip install tomli`."
            ) from e
    try:
        return tomllib.loads(content)
    except tomllib.TOMLDecodeError as e:
        raise CLIError(f"Invalid TOML in the script's PEP 723 header: {e}") from e


def _type_error(key: str, value: Any, expected: str) -> CLIError:
    return CLIError(f"'{key}' in the script's [{TABLE_NAME}] table must be {expected}, got: {value!r}.")


def _as_str(table: dict[str, Any], key: str) -> str | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _type_error(key, value, "a string")
    return value


def _as_timeout(table: dict[str, Any]) -> str | None:
    value = table.get("timeout")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise _type_error("timeout", value, 'a duration string (e.g. "30m") or a number of seconds')
    return str(value)


def _as_str_table(table: dict[str, Any], key: str) -> dict[str, str]:
    value = table.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _type_error(key, value, "a table of strings")
    for name, item in value.items():
        # Stringifying a bool or an int here would send a value the author never wrote.
        if not isinstance(item, str):
            raise _type_error(f"{key}.{name}", item, "a string")
    return value


def _as_str_list(table: dict[str, Any], key: str) -> list[str]:
    value = table.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise _type_error(key, value, "a list of strings")
    return value


def _as_secret_names(table: dict[str, Any]) -> list[str]:
    if isinstance(table.get("secrets"), dict):
        raise CLIError(
            f"'secrets' in the script's [{TABLE_NAME}] table must be a list of names (e.g. `secrets = [\"HF_TOKEN\"]`),"
            " not a table: scripts travel publicly, so secret values are always read from the environment of"
            " whoever runs the script."
        )
    return _as_str_list(table, "secrets")
