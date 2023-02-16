import unittest
from unittest.mock import Mock, call, patch

from huggingface_hub.utils._telemetry import send_telemetry
from .testing_constants import ENDPOINT_STAGING


@patch("huggingface_hub.utils._telemetry.requests.head")
class TestSendTelemetry(unittest.TestCase):
    def test_topic_normal(self, mock_request: Mock) -> None:
        send_telemetry(topic="examples")
        mock_request.assert_called_once()
        self.assertEqual(mock_request.call_args[0][0], f"{ENDPOINT_STAGING}/api/telemetry/examples")

    def test_topic_with_subtopic(self, mock_request: Mock) -> None:
        send_telemetry(topic="gradio/image/this_one")
        mock_request.assert_called_once()
        self.assertEqual(mock_request.call_args[0][0], f"{ENDPOINT_STAGING}/api/telemetry/gradio/image/this_one")

    def test_topic_quoted(self, mock_request: Mock) -> None:
        send_telemetry(topic="foo bar")
        mock_request.assert_called_once()
        self.assertEqual(mock_request.call_args[0][0], f"{ENDPOINT_STAGING}/api/telemetry/foo%20bar")

    @patch("huggingface_hub.utils._telemetry.constants.HF_HUB_OFFLINE", True)
    def test_hub_offline(self, mock_request: Mock) -> None:
        send_telemetry(topic="topic")
        mock_request.assert_not_called()

    @patch("huggingface_hub.utils._telemetry.constants.HF_HUB_DISABLE_TELEMETRY", True)
    def test_telemetry_disabled(self, mock_request: Mock) -> None:
        send_telemetry(topic="topic")
        mock_request.assert_not_called()

    @patch("huggingface_hub.utils._telemetry.build_hf_headers")
    def test_telemetry_use_build_hf_headers(self, mock_headers: Mock, mock_request: Mock) -> None:
        send_telemetry(topic="topic")
        mock_request.assert_called_once()
        mock_headers.assert_called_once()
        self.assertEqual(mock_request.call_args[1]["headers"], mock_headers.return_value)


@patch("huggingface_hub.utils._telemetry.requests.head", side_effect=Exception("whatever"))
class TestSendTelemetryConnectionError(unittest.TestCase):
    def test_telemetry_exception_silenced(self, mock_request: Mock) -> None:
        with self.assertLogs(logger="huggingface_hub.utils._telemetry", level="DEBUG") as captured:
            send_telemetry(topic="topic")

        # Assert debug message with traceback for debug purposes
        self.assertEqual(len(captured.output), 1)
        self.assertEqual(
            captured.output[0],
            "DEBUG:huggingface_hub.utils._telemetry:Error while sending telemetry: whatever",
        )
