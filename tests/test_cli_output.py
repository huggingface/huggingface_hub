# coding=utf-8
# Copyright 2026-present, the HuggingFace Inc. team.
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

import json

import pytest

from huggingface_hub.cli._cli_utils import OutputFormatWithAuto
from huggingface_hub.cli._output import Output, _format_table_cell_human, _to_header
from huggingface_hub.utils._detect_agent import _STANDARD_AGENT_VARS, _TOOL_AGENTS


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Deterministic baseline: no agent env vars, no ANSI colors."""
    all_vars = list(_STANDARD_AGENT_VARS)
    for env_vars, _ in _TOOL_AGENTS:
        all_vars.extend(env_vars)
    for var in all_vars:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_COLOR", "1")


def _make(mode: OutputFormatWithAuto) -> Output:
    o = Output()
    o.set_mode(mode)
    return o


HUMAN = OutputFormatWithAuto.human
AGENT = OutputFormatWithAuto.agent
JSON = OutputFormatWithAuto.json
QUIET = OutputFormatWithAuto.quiet


# =============================================================================
# set_mode
# =============================================================================


def test_auto_resolves_to_human():
    assert Output().mode == HUMAN


def test_auto_resolves_to_agent(monkeypatch):
    monkeypatch.setenv("AI_AGENT", "test")
    assert Output().mode == AGENT


def test_auto_resets_after_explicit():
    o = _make(QUIET)
    o.set_mode(OutputFormatWithAuto.auto)
    assert o.mode == HUMAN


# =============================================================================
# out.result()
# =============================================================================

# TODO: Add more cases when migrating commands
_RESULT_CASES = [
    pytest.param("Logged in", {"user": "Wauplin", "orgs": "org1,org2"}, id="whoami"),
]


@pytest.mark.parametrize(("message", "data"), _RESULT_CASES)
def test_result_human(capsys, message, data):
    _make(HUMAN).result(message, **data)
    out = capsys.readouterr().out
    assert "✓" in out
    assert message in out
    for k, v in data.items():
        assert f"  {k}: {v}" in out


@pytest.mark.parametrize(("message", "data"), _RESULT_CASES)
def test_result_agent(capsys, message, data):
    _make(AGENT).result(message, **data)
    out = capsys.readouterr().out.strip()
    for k, v in data.items():
        assert f"{k}={v}" in out


@pytest.mark.parametrize(("message", "data"), _RESULT_CASES)
def test_result_json(capsys, message, data):
    _make(JSON).result(message, **data)
    parsed = json.loads(capsys.readouterr().out)
    for k, v in data.items():
        assert parsed[k] == v


@pytest.mark.parametrize(("message", "data"), _RESULT_CASES)
def test_result_quiet(capsys, message, data):
    _make(QUIET).result(message, **data)
    out = capsys.readouterr().out.strip()
    assert out == str(list(data.values())[0])


def test_result_filters_none(capsys):
    _make(HUMAN).result("Done", keep="yes", drop=None)
    out = capsys.readouterr().out
    assert "keep" in out
    assert "drop" not in out


def test_result_agent_no_data(capsys):
    _make(AGENT).result("Logout successful")
    assert capsys.readouterr().out == "Logout successful\n"


# =============================================================================
# out.table()
# =============================================================================

_TABLE_CASES = [
    pytest.param(
        ["id", "downloads", "likes", "pipeline_tag"],
        [
            ["openai/gpt-oss-120b", 4133088, 4631, "text-generation"],
            ["CohereLabs/cohere-transcribe-03-2026", 58683, 670, "automatic-speech-recognition"],
        ],
        id="models",  # simulates `hf models ls`
    ),
]


@pytest.mark.parametrize(("headers", "rows"), _TABLE_CASES)
def test_table_human(capsys, headers, rows):
    _make(HUMAN).table(headers, rows)
    out = capsys.readouterr().out
    for h in headers:
        assert _to_header(h) in out


@pytest.mark.parametrize(("headers", "rows"), _TABLE_CASES)
def test_table_agent(capsys, headers, rows):
    _make(AGENT).table(headers, rows)
    lines = capsys.readouterr().out.strip().split("\n")
    assert lines[0] == "\t".join(headers)
    assert len(lines) == len(rows) + 1


@pytest.mark.parametrize(("headers", "rows"), _TABLE_CASES)
def test_table_json(capsys, headers, rows):
    _make(JSON).table(headers, rows)
    parsed = json.loads(capsys.readouterr().out)
    assert len(parsed) == len(rows)
    assert list(parsed[0].keys()) == headers


@pytest.mark.parametrize(("headers", "rows"), _TABLE_CASES)
def test_table_quiet(capsys, headers, rows):
    _make(QUIET).table(headers, rows)
    ids = capsys.readouterr().out.strip().split("\n")
    assert ids == [row[0] for row in rows]


def test_table_human_truncates(capsys):
    tags = "safetensors, qwen3_5, unsloth, qwen, qwen3.5, reasoning, chain-of-thought"
    _make(HUMAN).table(["id", "tags"], [["openai/gpt-oss-120b", tags]])
    assert "..." in capsys.readouterr().out


def test_table_agent_no_truncation(capsys):
    tags = "safetensors, qwen3_5, unsloth, qwen, qwen3.5, reasoning, chain-of-thought"
    _make(AGENT).table(["id", "tags"], [["openai/gpt-oss-120b", tags]])
    assert tags in capsys.readouterr().out


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (HUMAN, "No results found.\n"),
        (AGENT, "No results found.\n"),
        (JSON, "[]\n"),
        (QUIET, ""),
    ],
)
def test_table_empty(capsys, mode, expected):
    _make(mode).table(["id"], [])
    assert capsys.readouterr().out == expected


# =============================================================================
# out.dict()
# =============================================================================

_DICT_CASES = [
    pytest.param(
        {"id": "openai/gpt-oss-120b", "downloads": 4133088, "pipeline_tag": "text-generation"},
        id="models-info",  # simulates `hf models info`
    ),
]


@pytest.mark.parametrize("data", _DICT_CASES)
def test_dict_human(capsys, data):
    _make(HUMAN).dict(data)
    out = capsys.readouterr().out
    assert "\n" in out.strip()  # indented
    assert json.loads(out) == data


@pytest.mark.parametrize("data", _DICT_CASES)
def test_dict_agent(capsys, data):
    _make(AGENT).dict(data)
    out = capsys.readouterr().out
    assert "\n" not in out.strip()  # compact
    assert json.loads(out) == data


@pytest.mark.parametrize("data", _DICT_CASES)
def test_dict_json(capsys, data):
    _make(JSON).dict(data)
    out = capsys.readouterr().out
    assert "\n" not in out.strip()  # compact
    assert json.loads(out) == data


@pytest.mark.parametrize("data", _DICT_CASES)
def test_dict_quiet(capsys, data):
    _make(QUIET).dict(data)
    out = capsys.readouterr().out
    assert json.loads(out) == data


# =============================================================================
# out.text()
# =============================================================================


def test_text_human(capsys):
    _make(HUMAN).text("hello")
    assert capsys.readouterr().out == "hello\n"


def test_text_agent_strips_ansi(capsys):
    _make(AGENT).text("\033[1mbold\033[0m")
    assert capsys.readouterr().out == "bold\n"


def test_text_json_silent(capsys):
    _make(JSON).text("ignored")
    assert capsys.readouterr().out == ""


# =============================================================================
# warning / error / hint -> stderr in all modes
# =============================================================================


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (HUMAN, "  Warning: 3 files were ignored by .gitignore patterns"),  # indented, ANSI stripped by NO_COLOR
        (AGENT, "Warning: 3 files were ignored by .gitignore patterns"),  # plain text
        (JSON, "Warning: 3 files were ignored by .gitignore patterns"),
        (QUIET, "Warning: 3 files were ignored by .gitignore patterns"),
    ],
)
def test_warning(capsys, mode, expected):
    _make(mode).warning("3 files were ignored by .gitignore patterns")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.rstrip() == expected


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (HUMAN, "  Error: Not logged in"),  # indented, ANSI stripped by NO_COLOR
        (AGENT, "Error: Not logged in"),  # plain text
        (JSON, "Error: Not logged in"),
        (QUIET, "Error: Not logged in"),
    ],
)
def test_error(capsys, mode, expected):
    _make(mode).error("Not logged in")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.rstrip() == expected


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (HUMAN, "  Set HF_DEBUG=1 for full traceback."),  # indented, no "Hint:" prefix, ANSI stripped by NO_COLOR
        (AGENT, "Hint: Set HF_DEBUG=1 for full traceback."),  # with prefix
        (JSON, "Hint: Set HF_DEBUG=1 for full traceback."),
        (QUIET, "Hint: Set HF_DEBUG=1 for full traceback."),
    ],
)
def test_hint(capsys, mode, expected):
    _make(mode).hint("Set HF_DEBUG=1 for full traceback.")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.rstrip() == expected



# =============================================================================
# helpers
# =============================================================================


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        ("camelCase", "CAMEL_CASE"),  # camelCase → SCREAMING_SNAKE
        ("PascalCase", "PASCAL_CASE"),  # PascalCase → SCREAMING_SNAKE
        ("ID", "ID"),  # already uppercase, unchanged
    ],
)
def test_to_header(input, expected):
    assert _to_header(input) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("short", "short"),  # short string unchanged
        ("x" * 50, "x" * 32 + "..."),  # long string truncated at 35 chars
        (True, "✔"),  # bool True → checkmark
        (False, ""),  # bool False → empty
        (["a", "b"], "a, b"),  # list → comma-separated
    ],
)
def test_format_table_cell_human(value, expected):
    assert _format_table_cell_human(value) == expected
