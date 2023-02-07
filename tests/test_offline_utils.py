from io import BytesIO

import pytest
import requests

from huggingface_hub.file_download import http_get

from .testing_utils import (
    OfflineSimulationMode,
    RequestWouldHangIndefinitelyError,
    offline,
)


def test_offline_with_timeout():
    with offline(OfflineSimulationMode.CONNECTION_TIMES_OUT):
        with pytest.raises(RequestWouldHangIndefinitelyError):
            requests.request("GET", "https://huggingface.co")
        with pytest.raises(requests.exceptions.ConnectTimeout):
            requests.request("GET", "https://huggingface.co", timeout=1.0)
        with pytest.raises(requests.exceptions.ConnectTimeout):
            http_get("https://huggingface.co", BytesIO())


def test_offline_with_connection_error():
    with offline(OfflineSimulationMode.CONNECTION_FAILS):
        with pytest.raises(requests.exceptions.ConnectionError):
            requests.request("GET", "https://huggingface.co")
        with pytest.raises(requests.exceptions.ConnectionError):
            http_get("https://huggingface.co", BytesIO())


def test_offline_with_datasets_offline_mode_enabled():
    with offline(OfflineSimulationMode.HF_HUB_OFFLINE_SET_TO_1):
        with pytest.raises(ConnectionError):
            http_get("https://huggingface.co", BytesIO())
