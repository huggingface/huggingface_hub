import unittest
from unittest.mock import Mock, patch

from huggingface_hub.utils._errors import (
    BadRequestError,
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    _raise_convert_bad_request,
    _raise_for_status,
    _raise_with_request_id,
    hf_raise_for_status,
)
from requests.models import Response

from .testing_utils import expect_deprecation


class TestErrorUtils(unittest.TestCase):
    def test_hf_raise_for_status_repo_not_found(self) -> None:
        response = Response()
        response.headers = {"X-Error-Code": "RepoNotFound", "X-Request-Id": 123}
        response.status_code = 404
        with self.assertRaisesRegex(
            RepositoryNotFoundError, "Repository Not Found"
        ) as context:
            hf_raise_for_status(response)

        self.assertEqual(context.exception.response.status_code, 404)
        self.assertIn("Request ID: 123", str(context.exception))

    def test_hf_raise_for_status_repo_not_found_without_error_code(self) -> None:
        response = Response()
        response.headers = {"X-Request-Id": 123}
        response.status_code = 401
        with self.assertRaisesRegex(
            RepositoryNotFoundError, "Repository Not Found"
        ) as context:
            hf_raise_for_status(response)

        self.assertEqual(context.exception.response.status_code, 401)
        self.assertIn("Request ID: 123", str(context.exception))

    def test_hf_raise_for_status_revision_not_found(self) -> None:
        response = Response()
        response.headers = {"X-Error-Code": "RevisionNotFound", "X-Request-Id": 123}
        response.status_code = 404
        with self.assertRaisesRegex(
            RevisionNotFoundError, "Revision Not Found"
        ) as context:
            hf_raise_for_status(response)

        self.assertEqual(context.exception.response.status_code, 404)
        self.assertIn("Request ID: 123", str(context.exception))

    def test_hf_raise_for_status_entry_not_found(self) -> None:
        response = Response()
        response.headers = {"X-Error-Code": "EntryNotFound", "X-Request-Id": 123}
        response.status_code = 404
        with self.assertRaisesRegex(EntryNotFoundError, "Entry Not Found") as context:
            hf_raise_for_status(response)

        self.assertEqual(context.exception.response.status_code, 404)
        self.assertIn("Request ID: 123", str(context.exception))

    def test_hf_raise_for_status_bad_request_no_endpoint_name(self) -> None:
        """Test HTTPError converted to BadRequestError if error 400."""
        response = Response()
        response.status_code = 400
        with self.assertRaisesRegex(BadRequestError, "Bad request:") as context:
            hf_raise_for_status(response)
        self.assertEqual(context.exception.response.status_code, 400)

    def test_hf_raise_for_status_bad_request_with_endpoint_name(self) -> None:
        """Test endpoint name is added to BadRequestError message."""
        response = Response()
        response.status_code = 400
        with self.assertRaisesRegex(
            BadRequestError, "Bad request for preupload endpoint:"
        ) as context:
            hf_raise_for_status(response, endpoint_name="preupload")
        self.assertEqual(context.exception.response.status_code, 400)

    def test_hf_raise_for_status_fallback(self) -> None:
        """Test HTTPError is converted to HfHubHTTPError."""
        response = Response()
        response.status_code = 404
        response.headers = {
            "X-Request-Id": "test-id",
        }
        response.url = "test_URL"
        with self.assertRaisesRegex(HfHubHTTPError, "Request ID: test-id") as context:
            hf_raise_for_status(response)

        self.assertEqual(context.exception.response.status_code, 404)
        self.assertEqual(context.exception.response.url, "test_URL")

    @patch("huggingface_hub.utils._errors.hf_raise_for_status")
    def test_raise_for_status(self, mock_hf_raise_for_status: Mock) -> None:
        """Test `_raise_for_status` alias."""
        response_mock = Mock()
        _raise_for_status(response_mock)
        mock_hf_raise_for_status.assert_called_once_with(response_mock)

    @expect_deprecation("_raise_with_request_id")
    @patch("huggingface_hub.utils._errors.hf_raise_for_status")
    def test_raise_with_request_id(self, mock_hf_raise_for_status: Mock) -> None:
        """Test `_raise_with_request_id` alias."""
        response_mock = Mock()
        _raise_with_request_id(response_mock)
        mock_hf_raise_for_status.assert_called_once_with(response_mock)

    @expect_deprecation("_raise_convert_bad_request")
    @patch("huggingface_hub.utils._errors.hf_raise_for_status")
    def test_raise_convert_bad_request(self, mock_hf_raise_for_status: Mock) -> None:
        """Test `_raise_convert_bad_request` alias."""
        response_mock = Mock()
        endpoint_name_mock = Mock()
        _raise_convert_bad_request(response_mock, endpoint_name_mock)
        mock_hf_raise_for_status.assert_called_once_with(
            response_mock, endpoint_name=endpoint_name_mock
        )


class TestHfHubHTTPError(unittest.TestCase):
    response: Response

    def setUp(self) -> None:
        """Setup with a default response."""
        self.response = Response()
        self.response.status_code = 404
        self.response.url = "test_URL"

    def test_hf_hub_http_error_initialization(self) -> None:
        """Test HfHubHTTPError is initialized properly."""
        error = HfHubHTTPError("this is a message", response=self.response)
        self.assertEqual(str(error), "this is a message")
        self.assertEqual(error.response, self.response)
        self.assertIsNone(error.request_id)
        self.assertIsNone(error.server_message)

    def test_hf_hub_http_error_init_with_request_id(self) -> None:
        """Test request id is added to the message."""
        self.response.headers = {"X-Request-Id": "test-id"}
        error = HfHubHTTPError("this is a message", response=self.response)
        self.assertEqual(str(error), "this is a message (Request ID: test-id)")
        self.assertEqual(error.request_id, "test-id")

    def test_hf_hub_http_error_init_with_request_id_and_multiline_message(self) -> None:
        """Test request id is added to the end of the first line."""
        self.response.headers = {"X-Request-Id": "test-id"}
        error = HfHubHTTPError(
            "this is a message\nthis is more details", response=self.response
        )
        self.assertEqual(
            str(error), "this is a message (Request ID: test-id)\nthis is more details"
        )

        error = HfHubHTTPError(
            "this is a message\n\nthis is more details", response=self.response
        )
        self.assertEqual(
            str(error),
            "this is a message (Request ID: test-id)\n\nthis is more details",
        )

    def test_hf_hub_http_error_init_with_request_id_already_in_message(self) -> None:
        """Test request id is not duplicated in error message (case insensitive)"""
        self.response.headers = {"X-Request-Id": "test-id"}
        error = HfHubHTTPError(
            "this is a message on request TEST-ID", response=self.response
        )
        self.assertEqual(str(error), "this is a message on request TEST-ID")
        self.assertEqual(error.request_id, "test-id")

    def test_hf_hub_http_error_init_with_server_error(self) -> None:
        """Test server error is added to the error message."""
        self.response._content = (
            b'{"error": "This is a message returned by the server"}'
        )
        error = HfHubHTTPError("this is a message", response=self.response)
        self.assertEqual(
            str(error), "this is a message\n\nThis is a message returned by the server"
        )
        self.assertEqual(
            error.server_message, "This is a message returned by the server"
        )

    def test_hf_hub_http_error_init_with_server_error_and_multiline_message(
        self,
    ) -> None:
        """Test server error is added to the error message after the details."""
        self.response._content = (
            b'{"error": "This is a message returned by the server"}'
        )
        error = HfHubHTTPError(
            "this is a message\n\nSome details.", response=self.response
        )
        self.assertEqual(
            str(error),
            "this is a message\n\nSome details.\nThis is a message returned by the"
            " server",
        )

    def test_hf_hub_http_error_init_with_server_error_already_in_message(
        self,
    ) -> None:
        """Test server error is not duplicated if already in details.

        Case insensitive.
        """
        self.response._content = b'{"error": "repo NOT found"}'
        error = HfHubHTTPError(
            "this is a message\n\nRepo Not Found. and more\nand more",
            response=self.response,
        )
        self.assertEqual(
            str(error),
            "this is a message\n\nRepo Not Found. and more\nand more",
        )

    def test_hf_hub_http_error_init_with_unparsable_server_error(
        self,
    ) -> None:
        """Test error message is unchanged and exception is not raised.."""
        self.response._content = b"this is not a json-formatted string"
        error = HfHubHTTPError("this is a message", response=self.response)
        self.assertEqual(str(error), "this is a message")
        self.assertIsNone(error.server_message)  # still None since not parsed

    def test_hf_hub_http_error_append_to_message(self) -> None:
        """Test add extra information to existing HfHubHTTPError."""
        error = HfHubHTTPError("this is a message", response=self.response)
        error.args = error.args + (1, 2, 3)  # faking some extra args

        error.append_to_message("\nthis is an additional message")
        self.assertEqual(
            error.args,
            ("this is a message\nthis is an additional message", 1, 2, 3),
        )
        self.assertIsNone(error.server_message)  # added message is not from server
