import os
import shutil
from typing import Generator

import pytest
from _pytest.fixtures import SubRequest

import huggingface_hub
from huggingface_hub import constants
from huggingface_hub.utils import SoftTemporaryDirectory, logging

from .testing_utils import set_write_permission_and_retry


@pytest.fixture(autouse=True, scope="function")
def patch_constants(mocker):
    with SoftTemporaryDirectory() as cache_dir:
        mocker.patch.object(constants, "HF_HOME", cache_dir)
        mocker.patch.object(constants, "HF_HUB_CACHE", os.path.join(cache_dir, "hub"))
        mocker.patch.object(constants, "HF_XET_CACHE", os.path.join(cache_dir, "xet"))
        mocker.patch.object(constants, "HUGGINGFACE_HUB_CACHE", os.path.join(cache_dir, "hub"))
        mocker.patch.object(constants, "HF_ASSETS_CACHE", os.path.join(cache_dir, "assets"))
        mocker.patch.object(constants, "HF_TOKEN_PATH", os.path.join(cache_dir, "token"))
        mocker.patch.object(constants, "HF_STORED_TOKENS_PATH", os.path.join(cache_dir, "stored_tokens"))
        yield


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
        request.cls.cache_dir = cache_dir
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


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization", "user-agent", "cookie"],
        "ignore_localhost": True,
        "path_transformer": lambda path: path + ".yaml",
    }


@pytest.fixture(autouse=True)
def clear_lru_cache():
    from huggingface_hub.inference._providers.hf_inference import _check_supported_task

    _check_supported_task.cache_clear()
    yield
    _check_supported_task.cache_clear()
