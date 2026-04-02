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

import pytest

from huggingface_hub.cli._cli_utils import OutputFormatWithAuto
from huggingface_hub.cli._output import Output, _format_table_cell_human, _to_header
from huggingface_hub.utils._detect_agent import _STANDARD_AGENT_VARS, _TOOL_AGENTS


HUMAN = OutputFormatWithAuto.human
AGENT = OutputFormatWithAuto.agent
JSON = OutputFormatWithAuto.json
QUIET = OutputFormatWithAuto.quiet


def _normalize(text: str) -> list[str]:
    """
    Turn a triple-quoted string (or captured output) into comparable lines.
    """
    lines = text.split("\n")
    baseline = len(lines[-1]) - len(lines[-1].lstrip()) if lines else 0
    return [line[baseline:].rstrip() for line in lines if line.strip()]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Deterministic baseline: no agent env vars, no ANSI colors."""
    all_vars = list(_STANDARD_AGENT_VARS)
    for env_vars, _ in _TOOL_AGENTS:
        all_vars.extend(env_vars)
    for var in all_vars:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_COLOR", "1")


@pytest.fixture
def check(capsys):
    def _check(call, *, human, agent, json, quiet, stderr=False):
        failures = []
        for mode, expected in [(HUMAN, human), (AGENT, agent), (JSON, json), (QUIET, quiet)]:
            o = Output()
            o.set_mode(mode)
            call(o)
            captured = capsys.readouterr()
            actual_lines = _normalize(captured.err if stderr else captured.out)
            expected_lines = _normalize(expected)
            if actual_lines != expected_lines:
                failures.append(f"[{mode.value}] {actual_lines} != {expected_lines}")
        if failures:
            pytest.fail("\n".join(failures))

    return _check


# =============================================================================
# set_mode
# =============================================================================


def test_auto_resolves_to_human():
    assert Output().mode == HUMAN


def test_auto_resolves_to_agent(monkeypatch):
    monkeypatch.setenv("AI_AGENT", "test")
    assert Output().mode == AGENT


def test_auto_resets_after_explicit():
    o = Output()
    o.set_mode(QUIET)
    o.set_mode(OutputFormatWithAuto.auto)
    assert o.mode == HUMAN


# =============================================================================
# out.result()
# =============================================================================


def test_result(check):
    check(
        lambda out: out.result("Logged in", user="Wauplin", orgs="org1,org2", email=None),
        human="""
        ✓ Logged in
          user: Wauplin
          orgs: org1,org2
        """,
        agent="""
        user=Wauplin orgs=org1,org2
        """,
        json="""
        {"user": "Wauplin", "orgs": "org1,org2", "email": null}
        """,
        quiet="""
        Wauplin
        """,
    )


# =============================================================================
# out.table()
# =============================================================================


def test_table(check):
    headers = ["id", "downloads", "likes", "pipeline_tag"]
    rows = [
        ["openai/gpt-oss-120b", 4133088, 4631, "text-generation"],
        ["CohereLabs/cohere-transcribe-03-2026", 58683, 670, "automatic-speech-recognition"],
    ]
    check(
        lambda out: out.table(headers, rows),
        human="""
        ID                                  DOWNLOADS LIKES PIPELINE_TAG
        ----------------------------------- --------- ----- ----------------------------
        openai/gpt-oss-120b                 4133088   4631  text-generation
        CohereLabs/cohere-transcribe-03-... 58683     670   automatic-speech-recognition
        """,
        agent="""
        id\tdownloads\tlikes\tpipeline_tag
        openai/gpt-oss-120b\t4133088\t4631\ttext-generation
        CohereLabs/cohere-transcribe-03-2026\t58683\t670\tautomatic-speech-recognition
        """,
        json="""
        [{"id": "openai/gpt-oss-120b", "downloads": 4133088, "likes": 4631, "pipeline_tag": "text-generation"}, {"id": "CohereLabs/cohere-transcribe-03-2026", "downloads": 58683, "likes": 670, "pipeline_tag": "automatic-speech-recognition"}]
        """,
        quiet="""
        openai/gpt-oss-120b
        CohereLabs/cohere-transcribe-03-2026
        """,
    )


def test_table_empty(check):
    check(
        lambda out: out.table(["id"], []),
        human="""
        No results found.
        """,
        agent="""
        No results found.
        """,
        json="""
        []
        """,
        quiet="",
    )


# =============================================================================
# out.dict()
# =============================================================================


def test_dict(check):
    data = {"id": "openai/gpt-oss-120b", "downloads": 4133088, "pipeline_tag": "text-generation"}
    check(
        lambda out: out.dict(data),
        human="""
        {
          "id": "openai/gpt-oss-120b",
          "downloads": 4133088,
          "pipeline_tag": "text-generation"
        }
        """,
        agent="""
        {"id": "openai/gpt-oss-120b", "downloads": 4133088, "pipeline_tag": "text-generation"}
        """,
        json="""
        {"id": "openai/gpt-oss-120b", "downloads": 4133088, "pipeline_tag": "text-generation"}
        """,
        quiet="""
        {"id": "openai/gpt-oss-120b", "downloads": 4133088, "pipeline_tag": "text-generation"}
        """,
    )


# =============================================================================
# out.text()
# =============================================================================


def test_text(check):
    check(
        lambda out: out.text("hello"),
        human="""
        hello
        """,
        agent="""
        hello
        """,
        json="",
        quiet="",
    )


def test_text_with_agent_override(check):
    check(
        lambda out: out.text("Hello, Human!", agent="Hello, Agent!"),
        human="""
        Hello, Human!
        """,
        agent="""
        Hello, Agent!
        """,
        json="",
        quiet="",
    )


def test_text_agent_strips_ansi(check):
    check(
        lambda out: out.text("\033[1mbold\033[0m"),
        human="""
        \033[1mbold\033[0m
        """,
        agent="""
        bold
        """,
        json="",
        quiet="",
    )


# =============================================================================
# warning / error / hint -> stderr in all modes
# =============================================================================


def test_warning(check):
    check(
        lambda out: out.warning("3 files were ignored by .gitignore patterns"),
        stderr=True,
        human="""
        Warning: 3 files were ignored by .gitignore patterns
        """,
        agent="""
        Warning: 3 files were ignored by .gitignore patterns
        """,
        json="""
        Warning: 3 files were ignored by .gitignore patterns
        """,
        quiet="""
        Warning: 3 files were ignored by .gitignore patterns
        """,
    )


def test_error(check):
    check(
        lambda out: out.error("Not logged in"),
        stderr=True,
        human="""
        Error: Not logged in
        """,
        agent="""
        Error: Not logged in
        """,
        json="""
        Error: Not logged in
        """,
        quiet="""
        Error: Not logged in
        """,
    )


def test_hint(check):
    check(
        lambda out: out.hint("Set HF_DEBUG=1 for full traceback."),
        stderr=True,
        human="""
        Hint: Set HF_DEBUG=1 for full traceback.
        """,
        agent="""
        Hint: Set HF_DEBUG=1 for full traceback.
        """,
        json="""
        Hint: Set HF_DEBUG=1 for full traceback.
        """,
        quiet="""
        Hint: Set HF_DEBUG=1 for full traceback.
        """,
    )


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
