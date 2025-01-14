import inspect
import os
import shutil
import stat
import time
import unittest
import uuid
from contextlib import contextmanager
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Callable, Optional, Type, TypeVar, Union
from unittest.mock import Mock, patch

import pytest
import requests

from huggingface_hub.utils import (
    is_package_available,
    logging,
    reset_sessions,
)
from tests.testing_constants import ENDPOINT_PRODUCTION, ENDPOINT_PRODUCTION_URL_SCHEME


logger = logging.get_logger(__name__)

SMALL_MODEL_IDENTIFIER = "julien-c/bert-xsmall-dummy"
DUMMY_DIFF_TOKENIZER_IDENTIFIER = "julien-c/dummy-diff-tokenizer"
# Example model ids

# An actual model hosted on huggingface.co,
# w/ more details.
DUMMY_MODEL_ID = "julien-c/dummy-unknown"
DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT = "f2c752cfc5c0ab6f4bdec59acea69eefbee381c2"
# One particular commit (not the top of `main`)
DUMMY_MODEL_ID_REVISION_INVALID = "aaaaaaa"
# This commit does not exist, so we should 404.
DUMMY_MODEL_ID_PINNED_SHA1 = "d9e9f15bc825e4b2c9249e9578f884bbcb5e3684"
# Sha-1 of config.json on the top of `main`, for checking purposes
DUMMY_MODEL_ID_PINNED_SHA256 = "4b243c475af8d0a7754e87d7d096c92e5199ec2fe168a2ee7998e3b8e9bcb1d3"
# Sha-256 of pytorch_model.bin on the top of `main`, for checking purposes

# "hf-internal-testing/dummy-will-be-renamed" has been renamed to "hf-internal-testing/dummy-renamed"
DUMMY_RENAMED_OLD_MODEL_ID = "hf-internal-testing/dummy-will-be-renamed"
DUMMY_RENAMED_NEW_MODEL_ID = "hf-internal-testing/dummy-renamed"

SAMPLE_DATASET_IDENTIFIER = "lhoestq/custom_squad"
# Example dataset ids
DUMMY_DATASET_ID = "lhoestq/test"
DUMMY_DATASET_ID_REVISION_ONE_SPECIFIC_COMMIT = "81d06f998585f8ee10e6e3a2ea47203dc75f2a16"  # on branch "test-branch"

YES = ("y", "yes", "t", "true", "on", "1")
NO = ("n", "no", "f", "false", "off", "0")


def repo_name(id: Optional[str] = None, prefix: str = "repo") -> str:
    """
    Return a readable pseudo-unique repository name for tests.

    Example:
    ```py
    >>> repo_name()
    repo-2fe93f-16599646671840

    >>> repo_name("my-space", prefix='space')
    space-my-space-16599481979701
    """
    if id is None:
        id = uuid.uuid4().hex[:6]
    ts = int(time.time() * 10e3)
    return f"{prefix}-{id}-{ts}"


def parse_flag_from_env(key: str, default: bool = False) -> bool:
    try:
        value = os.environ[key]
    except KeyError:
        # KEY isn't set, default to `default`.
        return default

    # KEY is set, convert it to True or False.
    if value.lower() in YES:
        return True
    elif value.lower() in NO:
        return False
    else:
        # More values are supported, but let's keep the message simple.
        raise ValueError(f"If set, '{key}' must be one of {YES + NO}. Got '{value}'.")


def parse_int_from_env(key, default=None):
    try:
        value = os.environ[key]
    except KeyError:
        _value = default
    else:
        try:
            _value = int(value)
        except ValueError:
            raise ValueError("If set, {} must be a int.".format(key))
    return _value


_run_git_lfs_tests = parse_flag_from_env("RUN_GIT_LFS_TESTS", default=False)


def require_git_lfs(test_case):
    """
    Decorator to mark tests that requires git-lfs.

    git-lfs requires additional dependencies, and tests are skipped by default. Set the RUN_GIT_LFS_TESTS environment
    variable to a truthy value to run them.
    """
    if not _run_git_lfs_tests:
        return unittest.skip("test of git lfs workflow")(test_case)
    else:
        return test_case


def requires(package_name: str):
    """
    Decorator marking a test that requires PyTorch.
    These tests are skipped when PyTorch isn't installed.
    """

    def _inner(test_case):
        if not is_package_available(package_name):
            return unittest.skip(f"Test requires '{package_name}'")(test_case)
        else:
            return test_case

    return _inner


class RequestWouldHangIndefinitelyError(Exception):
    pass


class OfflineSimulationMode(Enum):
    CONNECTION_FAILS = 0
    CONNECTION_TIMES_OUT = 1
    HF_HUB_OFFLINE_SET_TO_1 = 2


@contextmanager
def offline(mode=OfflineSimulationMode.CONNECTION_FAILS, timeout=1e-16):
    """
    Simulate offline mode.

    There are three offline simulation modes:

    CONNECTION_FAILS (default mode): a ConnectionError is raised for each network call.
        Connection errors are created by mocking socket.socket
    CONNECTION_TIMES_OUT: the connection hangs until it times out.
        The default timeout value is low (1e-16) to speed up the tests.
        Timeout errors are created by mocking requests.request
    HF_HUB_OFFLINE_SET_TO_1: the HF_HUB_OFFLINE_SET_TO_1 environment variable is set to 1.
        This makes the http/ftp calls of the library instantly fail and raise an OfflineModeEnabled error.
    """
    import socket

    from requests import request as online_request

    def timeout_request(method, url, **kwargs):
        # Change the url to an invalid url so that the connection hangs
        invalid_url = "https://10.255.255.1"
        if kwargs.get("timeout") is None:
            raise RequestWouldHangIndefinitelyError(
                f"Tried a call to {url} in offline mode with no timeout set. Please set a timeout."
            )
        kwargs["timeout"] = timeout
        try:
            return online_request(method, invalid_url, **kwargs)
        except Exception as e:
            # The following changes in the error are just here to make the offline timeout error prettier
            e.request.url = url
            max_retry_error = e.args[0]
            max_retry_error.args = (max_retry_error.args[0].replace("10.255.255.1", f"OfflineMock[{url}]"),)
            e.args = (max_retry_error,)
            raise

    def offline_socket(*args, **kwargs):
        raise socket.error("Offline mode is enabled.")

    if mode is OfflineSimulationMode.CONNECTION_FAILS:
        # inspired from https://stackoverflow.com/a/18601897
        with patch("socket.socket", offline_socket):
            with patch("huggingface_hub.utils._http.get_session") as get_session_mock:
                with patch("huggingface_hub.file_download.get_session") as get_session_mock:
                    get_session_mock.return_value = requests.Session()  # not an existing one
                    yield
    elif mode is OfflineSimulationMode.CONNECTION_TIMES_OUT:
        # inspired from https://stackoverflow.com/a/904609
        with patch("requests.request", timeout_request):
            with patch("huggingface_hub.utils._http.get_session") as get_session_mock:
                with patch("huggingface_hub.file_download.get_session") as get_session_mock:
                    get_session_mock().request = timeout_request
                    yield
    elif mode is OfflineSimulationMode.HF_HUB_OFFLINE_SET_TO_1:
        with patch("huggingface_hub.constants.HF_HUB_OFFLINE", True):
            reset_sessions()
            yield
        reset_sessions()
    else:
        raise ValueError("Please use a value from the OfflineSimulationMode enum.")


def set_write_permission_and_retry(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def rmtree_with_retry(path: Union[str, Path]) -> None:
    shutil.rmtree(path, onerror=set_write_permission_and_retry)


def with_production_testing(func):
    file_download = patch("huggingface_hub.file_download.HUGGINGFACE_CO_URL_TEMPLATE", ENDPOINT_PRODUCTION_URL_SCHEME)
    hf_api = patch("huggingface_hub.constants.ENDPOINT", ENDPOINT_PRODUCTION)
    return hf_api(file_download(func))


def expect_deprecation(function_name: str):
    """
    Decorator to flag tests that we expect to use deprecated arguments.

    Args:
        function_name (`str`):
            Name of the function that we expect to use in a deprecated way.

    NOTE: if a test is expected to warns FutureWarnings but is not, the test will fail.

    Context: over time, some arguments/methods become deprecated. In order to track
             deprecation in tests, we run pytest with flag `-Werror::FutureWarning`.
             In order to keep old tests during the deprecation phase (before removing
             the feature completely) without changing them internally, we can flag
             them with this decorator.
    See full discussion in https://github.com/huggingface/huggingface_hub/pull/952.

    This decorator works hand-in-hand with the `_deprecate_arguments` and
    `_deprecate_positional_args` decorators.

    Example
    ```py
    # in src/hub_mixins.py
    from .utils._deprecation import _deprecate_arguments

    @_deprecate_arguments(version="0.12", deprecated_args={"repo_url"})
    def push_to_hub(...):
        (...)

    # in tests/test_something.py
    from .testing_utils import expect_deprecation

    class SomethingTest(unittest.TestCase):
        (...)

        @expect_deprecation("push_to_hub"):
        def test_push_to_hub_git_version(self):
            (...)
            push_to_hub(repo_url="something") <- Should warn with FutureWarnings
            (...)
    ```
    """

    def _inner_decorator(test_function: Callable) -> Callable:
        @wraps(test_function)
        def _inner_test_function(*args, **kwargs):
            with pytest.warns(FutureWarning, match=f".*'{function_name}'.*"):
                return test_function(*args, **kwargs)

        return _inner_test_function

    return _inner_decorator


def xfail_on_windows(reason: str, raises: Optional[Type[Exception]] = None):
    """
    Decorator to flag tests that we expect to fail on Windows.

    Will not raise an error if the expected error happens while running on Windows machine.
    If error is expected but does not happen, the test fails as well.

    Args:
        reason (`str`):
            Reason why it should fail.
        raises (`Type[Exception]`):
            The error type we except to happen.
    """

    def _inner_decorator(test_function: Callable) -> Callable:
        return pytest.mark.xfail(os.name == "nt", reason=reason, raises=raises, strict=True, run=True)(test_function)

    return _inner_decorator


T = TypeVar("T")


def handle_injection(cls: T) -> T:
    """Handle mock injection for each test of a test class.

    When patching variables on a class level, only relevant mocks will be injected to
    the tests. This has 2 advantages:
    1. There is no need to expect all mocks in test arguments when they are not needed.
    2. Default mock injection append all mocks 1 by 1 to the test args. If the order of
       the patch calls or test argument is changed, it can lead to unexpected behavior.

    NOTE: `@handle_injection` has to be defined after the `@patch` calls.

    Example:
    ```py
    @patch("something.foo")
    @patch("something_else.foo.bar") # order doesn't matter
    @handle_injection # after @patch calls
    def TestHelloWorld(unittest.TestCase):

        def test_hello_foo(self, mock_foo: Mock) -> None:
            (...)

        def test_hello_bar(self, mock_bar: Mock) -> None
            (...)

        def test_hello_both(self, mock_foo: Mock, mock_bar: Mock) -> None:
            (...)
    ```

    There are limitations with the current implementation:
    1. All patched variables must have different names.
       Named injection will not work with both `@patch("something.foo")` and
       `@patch("something_else.foo")` patches.
    2. Tests are expected to take only `self` and mock arguments. If it's not the case,
       this helper will fail.
    3. Tests arguments must follow the `mock_{variable_name}` naming.
       Example: `@patch("something._foo")` -> `"mock__foo"`.
    4. Tests arguments must be typed as `Mock`.

    If required, we can improve the current implementation in the future to mitigate
    those limitations.

    Based on:
    - https://stackoverflow.com/a/3467879
    - https://stackoverflow.com/a/30764825
    - https://stackoverflow.com/a/57115876

    NOTE: this decorator is inspired from the fixture system from pytest.
    """
    # Iterate over class functions and decorate tests
    # Taken from https://stackoverflow.com/a/3467879
    #        and https://stackoverflow.com/a/30764825
    for name, fn in inspect.getmembers(cls):
        if name.startswith("test_"):
            setattr(cls, name, handle_injection_in_test(fn))

    # Return decorated class
    return cls


def handle_injection_in_test(fn: Callable) -> Callable:
    """
    Handle injections at a test level. See `handle_injection` for more details.

    Example:
    ```py
    def TestHelloWorld(unittest.TestCase):

        @patch("something.foo")
        @patch("something_else.foo.bar") # order doesn't matter
        @handle_injection_in_test # after @patch calls
        def test_hello_foo(self, mock_foo: Mock) -> None:
            (...)
    ```
    """
    signature = inspect.signature(fn)
    parameters = signature.parameters

    @wraps(fn)
    def _inner(*args, **kwargs):
        assert kwargs == {}

        # Initialize new dict at least with `self`.
        assert len(args) > 0
        assert len(parameters) > 0
        new_kwargs = {"self": args[0]}

        # Check which mocks have been injected
        mocks = {}
        for value in args[1:]:
            assert isinstance(value, Mock)
            mock_name = "mock_" + value._extract_mock_name()
            mocks[mock_name] = value

        # Check which mocks are expected
        for name, parameter in parameters.items():
            if name == "self":
                continue
            assert parameter.annotation is Mock
            assert name in mocks, (
                f"Mock `{name}` not found for test `{fn.__name__}`. Available: {', '.join(sorted(mocks.keys()))}"
            )
            new_kwargs[name] = mocks[name]

        # Run test only with a subset of mocks
        return fn(**new_kwargs)

    return _inner


def use_tmp_repo(repo_type: str = "model") -> Callable[[T], T]:
    """
    Test decorator to create a repo for the test and properly delete it afterward.

    TODO: could we make `_api`, `_user` and `_token` cleaner ?

    Example:
    ```py
    from huggingface_hub import RepoUrl
    from .testing_utils import use_tmp_repo

    class HfApiCommonTest(unittest.TestCase):
        _api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)

        @use_tmp_repo()
        def test_create_tag_on_model(self, repo_url: RepoUrl) -> None:
            (...)

        @use_tmp_repo("dataset")
        def test_create_tag_on_dataset(self, repo_url: RepoUrl) -> None:
            (...)
    ```
    """

    def _inner_use_tmp_repo(test_fn: T) -> T:
        @wraps(test_fn)
        def _inner(*args, **kwargs):
            self = args[0]
            assert isinstance(self, unittest.TestCase)
            create_repo_kwargs = {}
            if repo_type == "space":
                create_repo_kwargs["space_sdk"] = "gradio"

            repo_url = self._api.create_repo(
                repo_id=repo_name(prefix=repo_type), repo_type=repo_type, **create_repo_kwargs
            )
            try:
                return test_fn(*args, **kwargs, repo_url=repo_url)
            finally:
                self._api.delete_repo(repo_id=repo_url.repo_id, repo_type=repo_type)

        return _inner

    return _inner_use_tmp_repo
