import unittest
from unittest.mock import Mock

import pytest
from httpx import Request, Response

from huggingface_hub.errors import (
    BadRequestError,
    DisabledRepoError,
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.utils._http import REPO_API_REGEX, X_AMZN_TRACE_ID, X_REQUEST_ID, _format, hf_raise_for_status


class TestErrorUtils(unittest.TestCase):
    def test_hf_raise_for_status_repo_not_found(self) -> None:
        response = Response(status_code=404, headers={"X-Error-Code": "RepoNotFound", X_REQUEST_ID: "123"})
        response.request = Request(method="GET", url="https://huggingface.co/fake")
        with self.assertRaisesRegex(RepositoryNotFoundError, "Repository Not Found") as context:
            hf_raise_for_status(response)

        assert context.exception.response.status_code == 404
        assert "Request ID: 123" in str(context.exception)

    def test_hf_raise_for_status_disabled_repo(self) -> None:
        response = Response(
            status_code=403, headers={"X-Error-Message": "Access to this resource is disabled.", X_REQUEST_ID: "123"}
        )
        response.request = Request(method="GET", url="https://huggingface.co/fake")

        with self.assertRaises(DisabledRepoError) as context:
            hf_raise_for_status(response)

        assert context.exception.response.status_code == 403
        assert "Request ID: 123" in str(context.exception)

    def test_hf_raise_for_status_401_repo_url_not_invalid_token(self) -> None:
        response = Response(status_code=401, headers={X_REQUEST_ID: "123"})
        response.request = Request(method="GET", url="https://huggingface.co/api/models/username/reponame")
        with self.assertRaisesRegex(RepositoryNotFoundError, "Repository Not Found") as context:
            hf_raise_for_status(response)

        assert context.exception.response.status_code == 401
        assert "Request ID: 123" in str(context.exception)

    def test_hf_raise_for_status_401_repo_url_invalid_token(self) -> None:
        response = Response(
            status_code=401,
            headers={X_REQUEST_ID: "123", "X-Error-Message": "Invalid credentials in Authorization header"},
        )
        response.request = Request(method="GET", url="https://huggingface.co/api/models/username/reponame")
        with self.assertRaisesRegex(HfHubHTTPError, "Invalid credentials in Authorization header") as context:
            hf_raise_for_status(response)

        assert context.exception.response.status_code == 401
        assert "Request ID: 123" in str(context.exception)

    def test_hf_raise_for_status_403_wrong_token_scope(self) -> None:
        response = Response(
            status_code=403, headers={X_REQUEST_ID: "123", "X-Error-Message": "specific error message"}
        )
        response.request = Request(method="GET", url="https://huggingface.co/api/repos/create")
        expected_message_part = "403 Forbidden: specific error message"
        with self.assertRaisesRegex(HfHubHTTPError, expected_message_part) as context:
            hf_raise_for_status(response)

        assert context.exception.response.status_code == 403
        assert "Request ID: 123" in str(context.exception)

    def test_hf_raise_for_status_401_not_repo_url(self) -> None:
        response = Response(status_code=401, headers={X_REQUEST_ID: "123"})
        response.request = Request(method="GET", url="https://huggingface.co/api/collections")
        with self.assertRaises(HfHubHTTPError) as context:
            hf_raise_for_status(response)

        assert context.exception.response.status_code == 401
        assert "Request ID: 123" in str(context.exception)

    def test_hf_raise_for_status_revision_not_found(self) -> None:
        response = Response(status_code=404, headers={"X-Error-Code": "RevisionNotFound", X_REQUEST_ID: "123"})
        response.request = Request(method="GET", url="https://huggingface.co/fake")
        with self.assertRaisesRegex(RevisionNotFoundError, "Revision Not Found") as context:
            hf_raise_for_status(response)

        assert context.exception.response.status_code == 404
        assert "Request ID: 123" in str(context.exception)

    def test_hf_raise_for_status_entry_not_found(self) -> None:
        response = Response(status_code=404, headers={"X-Error-Code": "EntryNotFound", X_REQUEST_ID: "123"})
        response.request = Request(method="GET", url="https://huggingface.co/fake")
        with self.assertRaisesRegex(EntryNotFoundError, "Entry Not Found") as context:
            hf_raise_for_status(response)

        assert context.exception.response.status_code == 404
        assert "Request ID: 123" in str(context.exception)

    def test_hf_raise_for_status_bad_request_no_endpoint_name(self) -> None:
        """Test HTTPError converted to BadRequestError if error 400."""
        response = Response(status_code=400)
        response.request = Request(method="GET", url="https://huggingface.co/fake")
        with self.assertRaisesRegex(BadRequestError, "Bad request:") as context:
            hf_raise_for_status(response)
        assert context.exception.response.status_code == 400

    def test_hf_raise_for_status_bad_request_with_endpoint_name(self) -> None:
        """Test endpoint name is added to BadRequestError message."""
        response = Response(status_code=400)
        response.request = Request(method="GET", url="https://huggingface.co/fake")
        with self.assertRaisesRegex(BadRequestError, "Bad request for preupload endpoint:") as context:
            hf_raise_for_status(response, endpoint_name="preupload")
        assert context.exception.response.status_code == 400

    def test_hf_raise_for_status_fallback(self) -> None:
        """Test HTTPError is converted to HfHubHTTPError."""
        response = Response(status_code=404, headers={X_REQUEST_ID: "test-id"})
        response.request = Request(method="GET", url="https://huggingface.co/fake")
        with self.assertRaisesRegex(HfHubHTTPError, "Request ID: test-id") as context:
            hf_raise_for_status(response)

        assert context.exception.response.status_code == 404
        assert context.exception.response.url == "https://huggingface.co/fake"


class TestHfHubHTTPError(unittest.TestCase):
    response: Response

    def setUp(self) -> None:
        """Setup with a default response."""
        self.response = Response(status_code=404, request=Request(method="GET", url="https://huggingface.co/fake"))

    def test_hf_hub_http_error_initialization(self) -> None:
        """Test HfHubHTTPError is initialized properly."""
        error = HfHubHTTPError("this is a message", response=self.response)
        assert str(error) == "this is a message"
        assert error.response == self.response
        assert error.request_id is None
        assert error.server_message is None

    def test_hf_hub_http_error_init_with_request_id(self) -> None:
        """Test request id is added to the message."""
        self.response.headers = {X_REQUEST_ID: "test-id"}
        error = _format(HfHubHTTPError, "this is a message", response=self.response)
        assert str(error) == "this is a message (Request ID: test-id)"
        assert error.request_id == "test-id"

    def test_hf_hub_http_error_init_with_request_id_and_multiline_message(self) -> None:
        """Test request id is added to the end of the first line."""
        self.response.headers = {X_REQUEST_ID: "test-id"}
        error = _format(HfHubHTTPError, "this is a message\nthis is more details", response=self.response)
        assert str(error) == "this is a message (Request ID: test-id)\nthis is more details"

        error = _format(HfHubHTTPError, "this is a message\n\nthis is more details", response=self.response)
        assert str(error) == "this is a message (Request ID: test-id)\n\nthis is more details"

    def test_hf_hub_http_error_init_with_request_id_already_in_message(self) -> None:
        """Test request id is not duplicated in error message (case-insensitive)"""
        self.response.headers = {X_REQUEST_ID: "test-id"}
        error = _format(HfHubHTTPError, "this is a message on request TEST-ID", response=self.response)
        assert str(error) == "this is a message on request TEST-ID"
        assert error.request_id == "test-id"

    def test_hf_hub_http_error_init_with_server_error(self) -> None:
        """Test server error is added to the error message."""
        self.response._content = b'{"error": "This is a message returned by the server"}'
        error = _format(HfHubHTTPError, "this is a message", response=self.response)
        assert str(error) == "this is a message\n\nThis is a message returned by the server"
        assert error.server_message == "This is a message returned by the server"

    def test_hf_hub_http_error_init_with_server_error_and_multiline_message(
        self,
    ) -> None:
        """Test server error is added to the error message after the details."""
        self.response._content = b'{"error": "This is a message returned by the server"}'
        error = _format(HfHubHTTPError, "this is a message\n\nSome details.", response=self.response)
        assert str(error) == "this is a message\n\nSome details.\nThis is a message returned by the server"

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
        error = _format(HfHubHTTPError, "this is a message\n\nSome details.", response=self.response)
        assert str(error) == "this is a message\n\nSome details.\nthis is error 1\nthis is error 2"

    def test_hf_hub_http_error_init_with_server_error_already_in_message(
        self,
    ) -> None:
        """Test server error is not duplicated if already in details.

        Case-insensitive.
        """
        self.response._content = b'{"error": "repo NOT found"}'
        error = _format(
            HfHubHTTPError,
            "this is a message\n\nRepo Not Found. and more\nand more",
            response=self.response,
        )
        assert str(error) == "this is a message\n\nRepo Not Found. and more\nand more"

    def test_hf_hub_http_error_init_with_unparsable_server_error(
        self,
    ) -> None:
        """Server returned a text message (not as JSON) => should be added to the exception."""
        self.response._content = b"this is not a json-formatted string"
        error = _format(HfHubHTTPError, "this is a message", response=self.response)
        assert str(error) == "this is a message\n\nthis is not a json-formatted string"
        assert error.server_message == "this is not a json-formatted string"

    def test_hf_hub_http_error_append_to_message(self) -> None:
        """Test add extra information to existing HfHubHTTPError."""
        error = _format(HfHubHTTPError, "this is a message", response=self.response)
        error.args = error.args + (1, 2, 3)  # faking some extra args

        error.append_to_message("\nthis is an additional message")
        assert error.args == ("this is a message\nthis is an additional message", 1, 2, 3)

        assert error.server_message is None  # added message is not from server

    def test_hf_hub_http_error_init_with_error_message_in_header(self) -> None:
        """Test server error from header is added to the error message."""
        self.response.headers = {"X-Error-Message": "Error message from headers."}
        error = _format(HfHubHTTPError, "this is a message", response=self.response)
        assert str(error) == "this is a message\n\nError message from headers."
        assert error.server_message == "Error message from headers."

    def test_hf_hub_http_error_init_with_error_message_from_header_and_body(
        self,
    ) -> None:
        """Test server error from header and from body are added to the error message."""
        self.response._content = b'{"error": "Error message from body."}'
        self.response.headers = {"X-Error-Message": "Error message from headers."}
        error = _format(HfHubHTTPError, "this is a message", response=self.response)
        assert str(error) == "this is a message\n\nError message from headers.\nError message from body."
        assert error.server_message == "Error message from headers.\nError message from body."

    def test_hf_hub_http_error_init_with_error_message_duplicated_in_header_and_body(
        self,
    ) -> None:
        """Test server error from header and from body are the same.

        Should not duplicate it in the raised `HfHubHTTPError`.
        """
        self.response._content = b'{"error": "Error message duplicated in headers and body."}'
        self.response.headers = {"X-Error-Message": "Error message duplicated in headers and body."}
        error = _format(HfHubHTTPError, "this is a message", response=self.response)
        assert str(error) == "this is a message\n\nError message duplicated in headers and body."
        assert error.server_message == "Error message duplicated in headers and body."

    def test_hf_hub_http_error_without_request_id_with_amzn_trace_id(self) -> None:
        """Test request id is not duplicated in error message (case-insensitive)"""
        self.response.headers = {X_AMZN_TRACE_ID: "test-trace-id"}
        error = _format(HfHubHTTPError, "this is a message", response=self.response)
        assert str(error) == "this is a message (Amzn Trace ID: test-trace-id)"
        assert error.request_id == "test-trace-id"

    def test_hf_hub_http_error_with_request_id_and_amzn_trace_id(self) -> None:
        """Test request id is not duplicated in error message (case-insensitive)"""
        self.response.headers = {X_AMZN_TRACE_ID: "test-trace-id", X_REQUEST_ID: "test-id"}
        error = _format(HfHubHTTPError, "this is a message", response=self.response)
        assert str(error) == "this is a message (Request ID: test-id)"
        assert error.request_id == "test-id"

    def test_hf_hub_error_reconstruction(self) -> None:
        """Test HfHubHTTPError is reconstructed properly."""
        from copy import deepcopy

        mock_response = Response(status_code=404, request=Request(method="GET", url="https://huggingface.co/fake"))
        error = HfHubHTTPError("this is a message", response=mock_response)
        copy_error = deepcopy(error)
        assert str(copy_error) == str(error)
        assert copy_error.request_id == error.request_id
        assert copy_error.server_message == error.server_message


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


def test_hf_hub_http_error_inherits_from_os_error() -> None:
    """Test HfHubHTTPError inherits from OSError."""
    with pytest.raises(OSError):
        raise HfHubHTTPError("this is a message", response=Mock())
