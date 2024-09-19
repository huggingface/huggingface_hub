import os
import shutil
from typing import Generator
from unittest.mock import patch

import pytest
from _pytest.fixtures import SubRequest

import huggingface_hub
from huggingface_hub.utils import SoftTemporaryDirectory, logging

from .testing_constants import ENDPOINT_STAGING, HF_PROFILES_PATH, HF_TOKEN_PATH
from .testing_utils import set_write_permission_and_retry


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


@pytest.fixture(scope="module", autouse=True)
def use_tmp_file_paths():
    """
    Fixture to temporarily override HF_TOKEN_PATH, HF_PROFILES_PATH, and ENDPOINT.

    This fixture patches the constants in the huggingface_hub module to use the
    specified paths and the staging endpoint. It also ensures that the files are
    deleted after all tests in the module are completed.
    """
    with patch.multiple(
        "huggingface_hub.constants",
        HF_TOKEN_PATH=HF_TOKEN_PATH,
        HF_PROFILES_PATH=HF_PROFILES_PATH,
        ENDPOINT=ENDPOINT_STAGING,
    ):
        yield
    # Remove the temporary files after all tests in the module are completed.
    for path in [HF_TOKEN_PATH, HF_PROFILES_PATH]:
        if os.path.exists(path):
            os.remove(path)
