import os
import shutil
import time
from functools import wraps
from pathlib import Path
from typing import Generator, List

import pytest
import requests
from _pytest.fixtures import SubRequest
from _pytest.python import Function as PytestFunction
from requests.exceptions import HTTPError

import huggingface_hub
from huggingface_hub import HfApi
from huggingface_hub.utils import SoftTemporaryDirectory, logging
from huggingface_hub.utils._typing import CallableT

from .testing_constants import ENDPOINT_PRODUCTION, PRODUCTION_TOKEN
from .testing_utils import repo_name, set_write_permission_and_retry


logger = logging.get_logger(__name__)


@pytest.fixture
def fx_cache_dir(request: SubRequest) -> Generator[None, None, None]:
    """Add a `cache_dir` attribute pointing to a temporary directory in tests.

    Example:
    ```py
    @pytest.mark.usefixtures("fx_cache_dir")
    class TestWithCache(unittest.TestCase):
        cache_dir: Path

        def test_cache_dir(self) -> None:
            self.assertTrue(self.cache_dir.is_dir())
    ```
    """
    with SoftTemporaryDirectory() as cache_dir:
        request.cls.cache_dir = Path(cache_dir).resolve()
        yield
        # TemporaryDirectory is not super robust on Windows when a git repository is
        # cloned in it. See https://www.scivision.dev/python-tempfile-permission-error-windows/.
        shutil.rmtree(cache_dir, onerror=set_write_permission_and_retry)


@pytest.fixture(autouse=True)
def disable_symlinks_on_windows_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSymlinkDict(dict):
        def __contains__(self, __o: object) -> bool:
            return True  # consider any `cache_dir` to be already checked

        def __getitem__(self, __key: str) -> bool:
            return False  # symlinks are never supported

    if os.name == "nt" and os.environ.get("DISABLE_SYMLINKS_IN_WINDOWS_TESTS"):
        monkeypatch.setattr(
            huggingface_hub.file_download,
            "_are_symlinks_supported_in_dir",
            FakeSymlinkDict(),
        )


@pytest.fixture(autouse=True)
def disable_experimental_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_DISABLE_EXPERIMENTAL_WARNING", True)


def retry_on_transient_error(fn: CallableT) -> CallableT:
    """
    Retry test if failure because of unavailable service, bad gateway or race condition.

    Tests are retried up to 10 times, waiting 5s between each try.
    """
    NUMBER_OF_TRIES = 10
    WAIT_TIME = 5
    HTTP_ERRORS = (502, 504)  # 502 Bad gateway (repo creation) or 504 Gateway timeout

    @wraps(fn)
    def _inner(*args, **kwargs):
        retry_count = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except HTTPError as e:
                if retry_count >= NUMBER_OF_TRIES:
                    raise
                if e.response.status_code in HTTP_ERRORS:
                    logger.info(
                        f"Attempt {retry_count} failed with a {e.response.status_code} error. Retrying new execution"
                        f" in {WAIT_TIME} second(s)..."
                    )
                else:
                    raise
            except requests.Timeout:
                if retry_count >= NUMBER_OF_TRIES:
                    raise
                logger.info(
                    f"HTTP Timeout while interacting with the Hub. Retrying new execution in {WAIT_TIME} second(s)..."
                )
            except OSError:
                if retry_count >= NUMBER_OF_TRIES:
                    raise
                logger.info(
                    "Race condition met where we tried to `clone` before fully deleting a repository. Retrying new"
                    f" execution in {WAIT_TIME} second(s)..."
                )
            time.sleep(WAIT_TIME)
            retry_count += 1

    return _inner


def pytest_collection_modifyitems(items: List[PytestFunction]):
    """Alter all tests to retry on transient errors.

    Note: equivalent to the previously used `@retry_endpoint` decorator, but tests do
          not have to be decorated individually anymore.
    """
    # called after collection is completed
    # you can modify the ``items`` list
    # see https://docs.pytest.org/en/7.3.x/how-to/writing_hook_functions.html
    for item in items:
        item.obj = retry_on_transient_error(item.obj)


@pytest.fixture
def fx_production_space(request: SubRequest) -> Generator[None, None, None]:
    """Add a `repo_id` attribute referencing a Space repo on the production Hub.

    Fully testing Spaces is not currently possible on staging so we need to use the production
    environment for it. Tests are skipped if we can't find a `HUGGINGFACE_PRODUCTION_USER_TOKEN`
    environment variable.

    Example:
    ```py
    @pytest.mark.usefixtures("fx_production_space")
    class TestSpaceAPI(unittest.TestCase):
        repo_id: str
        api: HfApi

        def test_space(self) -> None:
            api.repo_info(repo_id, repo_type="space")
    ```
    """
    # Check if production token exists
    if not PRODUCTION_TOKEN:
        pytest.skip("Skip Space tests. `HUGGINGFACE_PRODUCTION_USER_TOKEN` environment variable is not set.")

    # Generate repo id from prod token
    api = HfApi(token=PRODUCTION_TOKEN, endpoint=ENDPOINT_PRODUCTION)
    user = api.whoami()["name"]
    repo_id = f"{user}/{repo_name(prefix='tmp_test_space')}"
    request.cls.api = api
    request.cls.repo_id = repo_id

    # Create and clean space repo
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", private=True)
    api.upload_file(
        path_or_fileobj=_BASIC_APP_PY_TEMPLATE,
        repo_id=repo_id,
        repo_type="space",
        path_in_repo="app.py",
    )
    yield
    api.delete_repo(repo_id=repo_id, repo_type="space")


_BASIC_APP_PY_TEMPLATE = """
import gradio as gr


def greet(name):
    return "Hello " + name + "!!"

iface = gr.Interface(fn=greet, inputs="text", outputs="text")
iface.launch()
""".encode()
