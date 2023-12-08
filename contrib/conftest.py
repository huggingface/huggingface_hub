import time
import uuid
from typing import Generator

import pytest
from _pytest.monkeypatch import MonkeyPatch

from huggingface_hub import delete_repo


@pytest.fixture(scope="session")
def token() -> str:
    # Not critical, only usable on the sandboxed CI instance.
    return "hf_94wBhPGp6KrrTH3KDchhKpRxZwd6dmHWLL"


@pytest.fixture(scope="session")
def user() -> str:
    return "__DUMMY_TRANSFORMERS_USER__"


@pytest.fixture(autouse=True, scope="session")
def login_as_dummy_user(token: str, monkeypatch: MonkeyPatch) -> Generator:
    """Login with dummy user token."""
    monkeypatch.setenv("HF_TOKEN", token)


@pytest.fixture
def repo_name(request) -> None:
    """
    Return a readable pseudo-unique repository name for tests.

    Example: "repo-2fe93f-16599646671840"
    """
    prefix = request.module.__name__  # example: `test_timm`
    id = uuid.uuid4().hex[:6]
    ts = int(time.time() * 10e3)
    return f"repo-{prefix}-{id}-{ts}"


@pytest.fixture
def cleanup_repo(user: str, repo_name: str) -> None:
    """Delete the repo at the end of the tests.

    TODO: Adapt to handle `repo_type` as well
    """
    yield  # run test
    delete_repo(repo_id=f"{user}/{repo_name}")
