import re
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet


REPO_ROOT = Path(__file__).parent.parent
HUB_SETUP_PY = REPO_ROOT / "setup.py"
HF_CLI_SETUP_PY = REPO_ROOT / "utils" / "hf" / "setup.py"

_PYTHON_REQUIRES_REGEX = re.compile(r"""^\s*python_requires=["'](?P<spec>[^"']+)["']""", re.MULTILINE)


def _python_requires(setup_py: Path) -> str:
    match = _PYTHON_REQUIRES_REGEX.search(setup_py.read_text(encoding="utf-8"))
    assert match is not None, f"No `python_requires=...` found in '{setup_py}'."
    return match.group("spec")


@pytest.fixture(scope="module")
def hub_python_requires() -> str:
    return _python_requires(HUB_SETUP_PY)


@pytest.fixture(scope="module")
def hf_cli_python_requires() -> str:
    return _python_requires(HF_CLI_SETUP_PY)


def test_hf_cli_python_requires_matches_hub(hub_python_requires: str, hf_cli_python_requires: str) -> None:
    """`hf` must declare the same `python_requires` as `huggingface_hub`.

    `hf` only ships a console script that imports `huggingface_hub`, and it pins
    `huggingface_hub==<same version>`, so the two must agree on the supported interpreters.
    """
    assert SpecifierSet(hf_cli_python_requires) == SpecifierSet(hub_python_requires), (
        f"`hf` declares python_requires={hf_cli_python_requires!r} but `huggingface_hub` declares "
        f"{hub_python_requires!r}. Update `utils/hf/setup.py` to match the root `setup.py`."
    )


def test_hf_cli_declares_no_python_unsupported_by_hub(hub_python_requires: str, hf_cli_python_requires: str) -> None:
    """The user-visible failure, stated as behavior rather than as string equality.

    On an interpreter that `hf` accepts but `huggingface_hub` rejects, `pip install hf` silently resolves to a
    years-old `hf` release (the last one whose pinned hub version still supported that Python), and asking for
    the current one fails with `No matching distribution found for huggingface_hub==<version>`.
    """
    hub_spec = SpecifierSet(hub_python_requires)
    hf_cli_spec = SpecifierSet(hf_cli_python_requires)

    candidates = [f"3.{minor}" for minor in range(8, 20)]
    allowed_by_hf_cli_only = [
        version for version in candidates if hf_cli_spec.contains(version) and not hub_spec.contains(version)
    ]

    assert not allowed_by_hf_cli_only, (
        f"`hf` accepts Python {allowed_by_hf_cli_only} but `huggingface_hub` does not, so `hf`'s own "
        "`huggingface_hub=={version}` pin cannot be resolved on those interpreters."
    )
