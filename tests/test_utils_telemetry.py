import logging
from queue import Queue
from unittest.mock import Mock

import pytest

from huggingface_hub.utils._telemetry import send_telemetry

from .testing_constants import ENDPOINT_STAGING


class TestSendTelemetry:
    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        self.queue: Queue = Queue()
        mocker.patch("huggingface_hub.utils._telemetry._TELEMETRY_QUEUE", self.queue)
        mocker.patch("huggingface_hub.utils._telemetry._TELEMETRY_THREAD", None)

        get_session_mock = Mock()
        self.mock_head = get_session_mock().head
        mocker.patch("huggingface_hub.utils._telemetry.get_session", get_session_mock)
        yield

    def test_topic_normal(self) -> None:
        send_telemetry(topic="examples")
        self.queue.join()  # Wait for the telemetry tasks to be completed
        self.mock_head.assert_called_once()
        assert self.mock_head.call_args[0][0] == f"{ENDPOINT_STAGING}/api/telemetry/examples"

    def test_topic_multiple(self) -> None:
        send_telemetry(topic="example1")
        send_telemetry(topic="example2")
        send_telemetry(topic="example3")
        self.queue.join()  # Wait for the telemetry tasks to be completed

        assert self.mock_head.call_count == 3  # 3 calls and order is preserved
        assert self.mock_head.call_args_list[0][0][0] == f"{ENDPOINT_STAGING}/api/telemetry/example1"
        assert self.mock_head.call_args_list[1][0][0] == f"{ENDPOINT_STAGING}/api/telemetry/example2"
        assert self.mock_head.call_args_list[2][0][0] == f"{ENDPOINT_STAGING}/api/telemetry/example3"

    def test_topic_with_subtopic(self) -> None:
        send_telemetry(topic="gradio/image/this_one")
        self.queue.join()  # Wait for the telemetry tasks to be completed
        self.mock_head.assert_called_once()
        assert self.mock_head.call_args[0][0] == f"{ENDPOINT_STAGING}/api/telemetry/gradio/image/this_one"

    def test_topic_quoted(self) -> None:
        send_telemetry(topic="foo bar")
        self.queue.join()  # Wait for the telemetry tasks to be completed
        self.mock_head.assert_called_once()
        assert self.mock_head.call_args[0][0] == f"{ENDPOINT_STAGING}/api/telemetry/foo%20bar"

    def test_hub_offline(self, mocker) -> None:
        mocker.patch("huggingface_hub.utils._telemetry.constants.HF_HUB_OFFLINE", True)
        send_telemetry(topic="topic")
        assert self.queue.empty()  # no tasks
        self.mock_head.assert_not_called()

    def test_telemetry_disabled(self, mocker) -> None:
        mocker.patch("huggingface_hub.utils._telemetry.constants.HF_HUB_DISABLE_TELEMETRY", True)
        send_telemetry(topic="topic")
        assert self.queue.empty()  # no tasks
        self.mock_head.assert_not_called()

    def test_telemetry_use_build_hf_headers(self, mocker) -> None:
        mock_headers = mocker.patch("huggingface_hub.utils._telemetry.build_hf_headers")
        send_telemetry(topic="topic")
        self.queue.join()  # Wait for the telemetry tasks to be completed
        self.mock_head.assert_called_once()
        mock_headers.assert_called_once()
        assert self.mock_head.call_args[1]["headers"] == mock_headers.return_value


class TestSendTelemetryConnectionError:
    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        self.queue: Queue = Queue()
        mocker.patch("huggingface_hub.utils._telemetry._TELEMETRY_QUEUE", self.queue)
        mocker.patch("huggingface_hub.utils._telemetry._TELEMETRY_THREAD", None)

        get_session_mock = Mock()
        get_session_mock().head.side_effect = Exception("whatever")
        mocker.patch("huggingface_hub.utils._telemetry.get_session", get_session_mock)
        yield

    def test_telemetry_exception_silenced(self, caplog) -> None:
        with caplog.at_level(logging.DEBUG, logger="huggingface_hub.utils._telemetry"):
            send_telemetry(topic="topic")
            self.queue.join()

        records = [r for r in caplog.records if r.name.startswith("huggingface_hub")]

        # Assert debug message with traceback for debug purposes
        assert len(records) == 1
        assert records[0].levelname == "DEBUG"
        assert records[0].name == "huggingface_hub.utils._telemetry"
        assert records[0].getMessage() == "Error while sending telemetry: whatever"
