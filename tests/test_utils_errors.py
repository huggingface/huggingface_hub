import unittest

import pytest
from requests.models import PreparedRequest, Response

from huggingface_hub.utils._errors import (
    REPO_API_REGEX,
    BadRequestError,
    DisabledRepoError,
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    hf_raise_for_status,
)


class TestErrorUtils(unittest.TestCase):
    def test_hf_raise_for_status_repo_not_found(self) -> None:
        response = Response()
        response.headers = {"X-Error-Code": "RepoNotFound", "X-Request-Id": 123}
        response.status_code = 404
        with self.assertRaisesRegex(RepositoryNotFoundError, "Repository Not Found") as context:
            hf_raise_for_status(response)

        self.assertEqual(context.exception.response.status_code, 404)
        self.assertIn("Request ID: 123", str(context.exception))

    def test_hf_raise_for_status_disabled_repo(self) -> None:
        response = Response()
        response.headers = {"X-Error-Message": "Access to this resource is disabled.", "X-Request-Id": 123}

        response.status_code = 403
        with self.assertRaises(DisabledRepoError) as context:
            hf_raise_for_status(response)

        self.assertEqual(context.exception.response.status_code, 403)
        self.assertIn("Request ID: 123", str(context.exception))

    def test_hf_raise_for_status_401_repo_url(self) -> None:
        response = Response()
        response.headers = {"X-Request-Id": 123}
        response.status_code = 401
        response.request = PreparedRequest()
        response.request.url = "https://huggingface.co/api/models/username/reponame"
        with self.assertRaisesRegex(RepositoryNotFoundError, "Repository Not Found") as context:
            hf_raise_for_status(response)

        self.assertEqual(context.exception.response.status_code, 401)
        self.assertIn("Request ID: 123", str(context.exception))

    def test_hf_raise_for_status_403_wrong_token_scope(self) -> None:
        response = Response()
        response.headers = {"X-Request-Id": 123}
        response.status_code = 403
        response.request = PreparedRequest()
        response.request.url = "https://huggingface.co/api/repos/create"
        expected_message_part = "make sure you have a token with the `write` role"
        with self.assertRaisesRegex(BadRequestError, expected_message_part) as context:
            hf_raise_for_status(response)

        self.assertEqual(context.exception.response.status_code, 403)
        self.assertIn("Request ID: 123", str(context.exception))

    def test_hf_raise_for_status_401_not_repo_url(self) -> None:
        response = Response()
        response.headers = {"X-Request-Id": 123}
        response.status_code = 401
        response.request = PreparedRequest()
        response.request.url = "https://huggingface.co/api/collections"
        with self.assertRaises(HfHubHTTPError) as context:
            hf_raise_for_status(response)

        self.assertEqual(context.exception.response.status_code, 401)
        self.assertIn("Request ID: 123", str(context.exception))

    def test_hf_raise_for_status_revision_not_found(self) -> None:
        response = Response()
        response.headers = {"X-Error-Code": "RevisionNotFound", "X-Request-Id": 123}
        response.status_code = 404
        with self.assertRaisesRegex(RevisionNotFoundError, "Revision Not Found") as context:
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
        with self.assertRaisesRegex(BadRequestError, "Bad request for preupload endpoint:") as context:
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
        error = HfHubHTTPError("this is a message\nthis is more details", response=self.response)
        self.assertEqual(str(error), "this is a message (Request ID: test-id)\nthis is more details")

        error = HfHubHTTPError("this is a message\n\nthis is more details", response=self.response)
        self.assertEqual(
            str(error),
            "this is a message (Request ID: test-id)\n\nthis is more details",
        )

    def test_hf_hub_http_error_init_with_request_id_already_in_message(self) -> None:
        """Test request id is not duplicated in error message (case insensitive)"""
        self.response.headers = {"X-Request-Id": "test-id"}
        error = HfHubHTTPError("this is a message on request TEST-ID", response=self.response)
        self.assertEqual(str(error), "this is a message on request TEST-ID")
        self.assertEqual(error.request_id, "test-id")

    def test_hf_hub_http_error_init_with_server_error(self) -> None:
        """Test server error is added to the error message."""
        self.response._content = b'{"error": "This is a message returned by the server"}'
        error = HfHubHTTPError("this is a message", response=self.response)
        self.assertEqual(str(error), "this is a message\n\nThis is a message returned by the server")
        self.assertEqual(error.server_message, "This is a message returned by the server")

    def test_hf_hub_http_error_init_with_server_error_and_multiline_message(
        self,
    ) -> None:
        """Test server error is added to the error message after the details."""
        self.response._content = b'{"error": "This is a message returned by the server"}'
        error = HfHubHTTPError("this is a message\n\nSome details.", response=self.response)
        self.assertEqual(
            str(error),
            "this is a message\n\nSome details.\nThis is a message returned by the server",
        )

    def test_hf_hub_http_error_init_with_multiple_server_errors(
        self,
    ) -> None:
        """Test server errors are added to the error message after the details.

        Regression test for https://github.com/huggingface/huggingface_hub/issues/1114.
        """
        self.response._content = (
            b'{"httpStatusCode": 400, "errors": [{"message": "this is error 1", "type":'
            b' "error"}, {"message": "this is error 2", "type": "error"}]}'
        )
        error = HfHubHTTPError("this is a message\n\nSome details.", response=self.response)
        self.assertEqual(
            str(error),
            "this is a message\n\nSome details.\nthis is error 1\nthis is error 2",
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

    def test_hf_hub_http_error_init_with_error_message_in_header(self) -> None:
        """Test server error from header is added to the error message."""
        self.response.headers = {"X-Error-Message": "Error message from headers."}
        error = HfHubHTTPError("this is a message", response=self.response)
        self.assertEqual(str(error), "this is a message\n\nError message from headers.")
        self.assertEqual(error.server_message, "Error message from headers.")

    def test_hf_hub_http_error_init_with_error_message_from_header_and_body(
        self,
    ) -> None:
        """Test server error from header and from body are added to the error message."""
        self.response._content = b'{"error": "Error message from body."}'
        self.response.headers = {"X-Error-Message": "Error message from headers."}
        error = HfHubHTTPError("this is a message", response=self.response)
        self.assertEqual(
            str(error),
            "this is a message\n\nError message from headers.\nError message from body.",
        )
        self.assertEqual(
            error.server_message,
            "Error message from headers.\nError message from body.",
        )

    def test_hf_hub_http_error_init_with_error_message_duplicated_in_header_and_body(
        self,
    ) -> None:
        """Test server error from header and from body are the same.

        Should not duplicate it in the raised `HfHubHTTPError`.
        """
        self.response._content = b'{"error": "Error message duplicated in headers and body."}'
        self.response.headers = {"X-Error-Message": "Error message duplicated in headers and body."}
        error = HfHubHTTPError("this is a message", response=self.response)
        self.assertEqual(
            str(error),
            "this is a message\n\nError message duplicated in headers and body.",
        )
        self.assertEqual(error.server_message, "Error message duplicated in headers and body.")


@pytest.mark.parametrize(
    ("url", "should_match"),
    [
        # Listing endpoints => False
        ("https://huggingface.co/api/models", False),
        ("https://huggingface.co/api/datasets", False),
        ("https://huggingface.co/api/spaces", False),
        # Create repo endpoint => False
        ("https://huggingface.co/api/repos/create", False),
        # Collection endpoints => False
        ("https://huggingface.co/api/collections", False),
        ("https://huggingface.co/api/collections/foo/bar", False),
        # Repo endpoints => True
        ("https://huggingface.co/api/models/repo_id", True),
        ("https://huggingface.co/api/datasets/repo_id", True),
        ("https://huggingface.co/api/spaces/repo_id", True),
        ("https://huggingface.co/api/models/username/repo_name/refs/main", True),
        ("https://huggingface.co/api/datasets/username/repo_name/refs/main", True),
        ("https://huggingface.co/api/spaces/username/repo_name/refs/main", True),
        # Inference Endpoint => False
        ("https://api.endpoints.huggingface.cloud/v2/endpoint/namespace", False),
        # Staging Endpoint => True
        ("https://hub-ci.huggingface.co/api/models/repo_id", True),
        ("https://hub-ci.huggingface.co/api/datasets/repo_id", True),
        ("https://hub-ci.huggingface.co/api/spaces/repo_id", True),
        # /resolve Endpoint => True
        ("https://huggingface.co/gpt2/resolve/main/README.md", True),
        ("https://huggingface.co/datasets/google/fleurs/resolve/revision/README.md", True),
        # Regression tests
        ("https://huggingface.co/bert-base/resolve/main/pytorch_model.bin", True),
        ("https://hub-ci.huggingface.co/__DUMMY_USER__/repo-1470b5/resolve/main/file.txt", True),
    ],
)
def test_repo_api_regex(url: str, should_match: bool) -> None:
    """Test the regex used to match repo API URLs."""
    if should_match:
        assert REPO_API_REGEX.match(url)
    else:
        assert REPO_API_REGEX.match(url) is None
