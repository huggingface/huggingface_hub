import os
import shutil
from contextlib import ExitStack
from typing import Generator, Protocol

import pytest
from _pytest.fixtures import SubRequest
from _pytest.skipping import evaluate_skip_marks

import huggingface_hub
from huggingface_hub import HfApi, RepoUrl, constants
from huggingface_hub.utils import SoftTemporaryDirectory, _detect_agent, logging
from huggingface_hub.utils._runtime import is_package_available

from .testing_constants import (
    ENDPOINT_PRODUCTION,
    ENDPOINT_PRODUCTION_URL_SCHEME,
    ENDPOINT_STAGING,
    TOKEN,
)
from .testing_utils import parse_flag_from_env, repo_name, set_write_permission_and_retry


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


@pytest.fixture(autouse=True)
def reset_oauth_refresh_state():
    """`get_token()` caches its OAuth refresh decision in process-global state: reset between tests."""
    from huggingface_hub.utils import _auth

    _auth._OAUTH_REFRESH_CACHE = None
    _auth._OAUTH_REFRESH_WARNED = False
    yield


@pytest.fixture(autouse=True)
def xet_mode(request: SubRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make Xet usage explicit and deterministic, locally and in CI.

    Three modes:
    - `@pytest.mark.xet`: test requires `hf_xet` => skipped when it is not installed,
      Xet force-enabled otherwise.
    - `@pytest.mark.no_xet`: test must run without `hf_xet` (e.g. legacy LFS behavior)
      => skipped when it is installed.
    - unmarked: test must work regardless of Xet => nothing is forced; the test runs
      with whatever the environment provides. CI runs unmarked tests both with and
      without `hf_xet` installed.
    """
    xet = request.node.get_closest_marker("xet") is not None
    no_xet = request.node.get_closest_marker("no_xet") is not None
    if xet and no_xet:
        pytest.fail("A test cannot be marked with both `xet` and `no_xet`.")
    if xet:
        if not is_package_available("hf_xet"):
            pytest.skip("Test requires `hf_xet` (marked with `pytest.mark.xet`)")
        monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", False)
        monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    elif no_xet:
        if is_package_available("hf_xet"):
            pytest.skip("Test must run without `hf_xet` installed (marked with `pytest.mark.no_xet`)")


@pytest.fixture(autouse=True)
def _clean_cli_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic baseline: agent detection disabled, no ANSI colors, reset output mode, fixed terminal width."""
    # Pin an empty agent registry so detection never hits the network and always reports "no agent",
    # regardless of any agent env var that may be set on the host running the tests.
    monkeypatch.setattr(_detect_agent, "_registry", _detect_agent._EMPTY_REGISTRY)
    monkeypatch.setenv("NO_COLOR", "1")
    # Pin terminal width so adaptive truncation produces deterministic output
    # regardless of the runner's actual terminal size. `shutil.get_terminal_size`
    # honors `$COLUMNS` first.
    monkeypatch.setenv("COLUMNS", "200")
    from huggingface_hub.cli._output import out

    out.set_mode()
    out.set_no_truncate(False)


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


@pytest.fixture(autouse=True)
def production_endpoint_marker(request: SubRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the Hub at production for tests marked with `@pytest.mark.production`.

    Mirrors the former `with_production_testing` decorator: patches both `ENDPOINT` and the
    download URL template so that bare `HfApi()` instances (and downloads) target production.
    The `api_prod` fixture passes the production endpoint explicitly and does not rely on this.
    """
    if request.node.get_closest_marker("production") is None:
        return
    monkeypatch.setattr(constants, "ENDPOINT", ENDPOINT_PRODUCTION)
    monkeypatch.setattr(constants, "HUGGINGFACE_CO_URL_TEMPLATE", ENDPOINT_PRODUCTION_URL_SCHEME)


@pytest.fixture(autouse=True)
def expect_deprecation_marker(request: SubRequest) -> Generator[None, None, None]:
    """Assert that a test marked `@pytest.mark.deprecated(...)` emits the expected `FutureWarning`s.

    Each argument is a function name; the test must emit a `FutureWarning` mentioning it (the suite runs
    with `-Werror::FutureWarning`, so an unraised warning fails the test). Example:
    ```py
    @pytest.mark.deprecated("duplicate_space", "duplicate_repo")
    def test_duplicate_space(...): ...
    ```
    """
    function_names = [name for marker in request.node.iter_markers("deprecated") for name in marker.args]
    # If the test is statically skipped (`@pytest.mark.skip`/`skipif`), its body never runs and no
    # warning is emitted; entering `pytest.warns` would then fail the item at teardown instead of
    # reporting it as skipped. `evaluate_skip_marks` returns a truthy result only when it will skip.
    if not function_names or evaluate_skip_marks(request.node):
        yield
        return
    with ExitStack() as stack:
        for function_name in function_names:
            stack.enter_context(pytest.warns(FutureWarning, match=rf".*'{function_name}'.*"))
        yield


@pytest.fixture(autouse=True)
def git_lfs_marker(request: SubRequest) -> None:
    """Skip tests marked `@pytest.mark.git_lfs` unless `RUN_GIT_LFS_TESTS` is truthy (git-lfs needed)."""
    if request.node.get_closest_marker("git_lfs") is not None and not parse_flag_from_env("RUN_GIT_LFS_TESTS"):
        pytest.skip("git-lfs test (set RUN_GIT_LFS_TESTS=1 to run)")


@pytest.fixture(scope="session")
def api() -> HfApi:
    """A staging `HfApi` client authenticated with the CI token."""
    return HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)


@pytest.fixture(scope="session")
def api_prod() -> HfApi:
    """An unauthenticated `HfApi` client targeting the production Hub (read-only tests)."""
    return HfApi(endpoint=ENDPOINT_PRODUCTION)


class RepoFactory(Protocol):
    """Type of the `repo_factory` fixture: create a temporary repo on staging and return its `RepoUrl`.

    The created repo is automatically deleted when the test ends.
    """

    def __call__(self, repo_type: str = "model", **kwargs) -> RepoUrl: ...


@pytest.fixture
def repo_factory(api: HfApi) -> Generator[RepoFactory, None, None]:
    """Create temporary repos on staging and delete them automatically when the test ends.

    Replaces the former `@use_tmp_repo` decorator. Example:
    ```py
    def test_something(api: HfApi, repo_factory):
        repo_url = repo_factory()              # a model repo
        dataset_url = repo_factory("dataset")  # a dataset repo
    ```
    """
    created: list[tuple[str, str]] = []

    def _factory(repo_type: str = "model", **kwargs) -> RepoUrl:
        if repo_type == "space" and "space_sdk" not in kwargs:
            kwargs["space_sdk"] = "gradio"
        repo_url = api.create_repo(repo_id=repo_name(prefix=repo_type), repo_type=repo_type, **kwargs)
        created.append((repo_url.repo_id, repo_type))
        return repo_url

    yield _factory

    for repo_id, repo_type in created:
        try:
            api.delete_repo(repo_id=repo_id, repo_type=repo_type)
        except Exception:
            pass
