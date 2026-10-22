import importlib.metadata
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from huggingface_hub.utils import _runtime
from huggingface_hub.utils._runtime import is_google_colab, is_notebook


class TestRuntimeUtils:
    def test_is_notebook(self) -> None:
        """Test `is_notebook`."""
        assert not is_notebook()

    def test_is_google_colab(self) -> None:
        """Test `is_google_colab`."""
        assert not is_google_colab()


def test_get_version_concurrent_first_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent first lookups must all see the installed version, not a "N/A" placeholder.

    Regression test for https://github.com/huggingface/huggingface_hub/issues/5101.
    """
    num_threads = 8
    monkeypatch.setattr(_runtime, "_package_versions", {})

    def slow_version(distribution_name: str) -> str:
        time.sleep(0.2)
        return "1.2.3"

    monkeypatch.setattr(importlib.metadata, "version", slow_version)
    barrier = threading.Barrier(num_threads)

    def get_version(_: int) -> str:
        barrier.wait()
        return _runtime._get_version("hf_xet")

    with ThreadPoolExecutor(num_threads) as pool:
        versions = list(pool.map(get_version, range(num_threads)))

    assert versions == ["1.2.3"] * num_threads
