from io import BytesIO

import httpx
import pytest

from huggingface_hub.file_download import http_get

from .testing_utils import OfflineSimulationMode, RequestWouldHangIndefinitelyError, offline


def test_offline_with_timeout():
    with offline(OfflineSimulationMode.CONNECTION_TIMES_OUT):
        with pytest.raises(RequestWouldHangIndefinitelyError):
            httpx.request("GET", "https://huggingface.co")
        with pytest.raises(httpx.ConnectTimeout):
            httpx.request("GET", "https://huggingface.co", timeout=1.0)
        with pytest.raises(httpx.ConnectTimeout):
            http_get("https://huggingface.co", BytesIO())


def test_offline_with_connection_error():
    with offline(OfflineSimulationMode.CONNECTION_FAILS):
        with pytest.raises(httpx.ConnectError):
            httpx.request("GET", "https://huggingface.co")
        with pytest.raises(httpx.ConnectError):
            http_get("https://huggingface.co", BytesIO())


def test_offline_with_datasets_offline_mode_enabled():
    with offline(OfflineSimulationMode.HF_HUB_OFFLINE_SET_TO_1):
        from huggingface_hub.errors import OfflineModeIsEnabled

        with pytest.raises(OfflineModeIsEnabled):
            http_get("https://huggingface.co", BytesIO())
