import os
import shutil
from pathlib import Path
from typing import Generator

import pytest
from _pytest.fixtures import SubRequest

import huggingface_hub
from huggingface_hub import HfApi, HfFolder
from huggingface_hub.utils import SoftTemporaryDirectory

from .testing_constants import ENDPOINT_PRODUCTION, PRODUCTION_TOKEN
from .testing_utils import repo_name, set_write_permission_and_retry


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


@pytest.fixture(autouse=True, scope="session")
def clean_hf_folder_token_for_tests() -> Generator:
    """Clean token stored on machine before all tests and reset it back at the end.

    Useful to avoid token deletion when running tests locally.
    """
    # Remove registered token
    token = HfFolder().get_token()
    HfFolder().delete_token()

    yield  # Run all tests

    # Set back token once all tests have passed
    if token is not None:
        HfFolder().save_token(token)


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
    if PRODUCTION_TOKEN is None:
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
