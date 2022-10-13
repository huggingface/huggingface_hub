import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator

import pytest

import huggingface_hub
from _pytest.fixtures import SubRequest
from huggingface_hub import HfFolder


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
    with TemporaryDirectory() as cache_dir:
        request.cls.cache_dir = Path(cache_dir).resolve()
        yield


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

    if os.name == "nt" and os.environ.get("RUNNING_ON_WINDOWS_CI_WITHOUT_SYMLINKS"):
        monkeypatch.setattr(
            huggingface_hub.file_download,
            "_are_symlinks_supported_in_dir",
            FakeSymlinkDict(),
        )
