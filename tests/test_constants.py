import json
import os
import subprocess
import sys

import pytest


@pytest.fixture
def read_constants(monkeypatch, tmp_path):
    # Constants are evaluated at import time. Use a subprocess to avoid changing
    # the constants patched by the shared test fixtures (including staging mode).
    for name in (
        "HUGGINGFACE_CO_STAGING",
        "HF_HOME",
        "HF_TOKEN_PATH",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_ENDPOINT",
        "HF_HUB_ETAG_TIMEOUT",
        "HF_HUB_DOWNLOAD_TIMEOUT",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    def read(*, check_token=False):
        script = """
import json
from huggingface_hub import HfApi, constants, get_token, hf_hub_download

assert HfApi().endpoint == constants.ENDPOINT
print(json.dumps({name: getattr(constants, name) for name in (
    "HF_HOME", "HF_HUB_CACHE", "HF_TOKEN_PATH", "HF_STORED_TOKENS_PATH",
    "ENDPOINT", "HF_HUB_ETAG_TIMEOUT", "HF_HUB_DOWNLOAD_TIMEOUT",
)}))
"""
        if check_token:
            script += "\nassert get_token() is None\n"
        return json.loads(subprocess.check_output([sys.executable, "-c", script], cwd=tmp_path, text=True))

    return read


@pytest.mark.parametrize(
    "name",
    [None, "HF_HOME", "HF_TOKEN_PATH", "HF_ENDPOINT", "HF_HUB_ETAG_TIMEOUT", "HF_HUB_DOWNLOAD_TIMEOUT"],
)
def test_empty_environment_uses_defaults(name, read_constants, monkeypatch, tmp_path):
    if name is not None:
        monkeypatch.setenv(name, "")

    assert read_constants(check_token=True) == {
        "HF_HOME": str(tmp_path / "huggingface"),
        "HF_HUB_CACHE": str(tmp_path / "huggingface" / "hub"),
        "HF_TOKEN_PATH": str(tmp_path / "huggingface" / "token"),
        "HF_STORED_TOKENS_PATH": str(tmp_path / "huggingface" / "stored_tokens"),
        "ENDPOINT": "https://huggingface.co",
        "HF_HUB_ETAG_TIMEOUT": 10,
        "HF_HUB_DOWNLOAD_TIMEOUT": 10,
    }


@pytest.mark.parametrize("path", ["relative", "~/custom-hf", "$XDG_CACHE_HOME/custom-hf"])
@pytest.mark.parametrize("timeout", ["25", "0"])
def test_non_empty_environment_is_preserved(path, timeout, read_constants, monkeypatch):
    monkeypatch.setenv("HF_HOME", path)
    monkeypatch.setenv("HF_TOKEN_PATH", os.path.join(path, "my-token"))
    monkeypatch.setenv("HF_ENDPOINT", "https://custom.huggingface.co/")
    monkeypatch.setenv("HF_HUB_ETAG_TIMEOUT", timeout)
    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", timeout)

    expanded_path = os.path.expandvars(os.path.expanduser(path))
    assert read_constants() == {
        "HF_HOME": expanded_path,
        "HF_HUB_CACHE": os.path.join(expanded_path, "hub"),
        "HF_TOKEN_PATH": os.path.join(expanded_path, "my-token"),
        "HF_STORED_TOKENS_PATH": os.path.join(expanded_path, "stored_tokens"),
        "ENDPOINT": "https://custom.huggingface.co",
        "HF_HUB_ETAG_TIMEOUT": 25 if timeout == "25" else 10,
        "HF_HUB_DOWNLOAD_TIMEOUT": 25 if timeout == "25" else 10,
    }
