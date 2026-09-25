from io import BytesIO

import httpx2
import pytest

from huggingface_hub.file_download import http_get

from .testing_utils import OfflineSimulationMode, RequestWouldHangIndefinitelyError, offline


def test_offline_with_timeout():
    with offline(OfflineSimulationMode.CONNECTION_TIMES_OUT):
        with pytest.raises(RequestWouldHangIndefinitelyError):
            httpx2.request("GET", "https://huggingface.co")
        with pytest.raises(httpx2.ConnectTimeout):
            httpx2.request("GET", "https://huggingface.co", timeout=1.0)
        with pytest.raises(httpx2.ConnectTimeout):
            http_get("https://huggingface.co", BytesIO())


def test_offline_with_connection_error():
    with offline(OfflineSimulationMode.CONNECTION_FAILS):
        with pytest.raises(httpx2.ConnectError):
            httpx2.request("GET", "https://huggingface.co")
        with pytest.raises(httpx2.ConnectError):
            http_get("https://huggingface.co", BytesIO())


def test_offline_with_datasets_offline_mode_enabled():
    with offline(OfflineSimulationMode.HF_HUB_OFFLINE_SET_TO_1):
        from huggingface_hub.errors import OfflineModeIsEnabled

        with pytest.raises(OfflineModeIsEnabled):
            http_get("https://huggingface.co", BytesIO())
