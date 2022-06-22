import unittest

from huggingface_hub.utils._errors import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    _raise_for_status,
    _raise_with_request_id,
)
from requests.exceptions import HTTPError
from requests.models import Response


class TestErrorUtils(unittest.TestCase):
    _raise_with_request_id

    def test__raise_for_status_repo_not_found(self):
        response = Response()
        response.headers = {
            "X-Error-Code": "RepoNotFound",
        }
        response.status_code = 404
        with self.assertRaisesRegex(
            RepositoryNotFoundError, "Repository Not Found"
        ) as context:
            _raise_for_status(response)

        self.assertEqual(context.exception.response.status_code, 404)

    def test__raise_for_status_repo_not_found_without_error_code(self):
        response = Response()
        response.status_code = 401
        with self.assertRaisesRegex(
            RepositoryNotFoundError, "Repository Not Found"
        ) as context:
            _raise_for_status(response)

        self.assertEqual(context.exception.response.status_code, 401)

    def test_raise_for_status_revision_not_found(self):
        response = Response()
        response.headers = {
            "X-Error-Code": "RevisionNotFound",
        }
        response.status_code = 404
        with self.assertRaisesRegex(
            RevisionNotFoundError, "Revision Not Found"
        ) as context:
            _raise_for_status(response)

        self.assertEqual(context.exception.response.status_code, 404)

    def test_raise_for_status_entry_not_found(self):
        response = Response()
        response.headers = {
            "X-Error-Code": "EntryNotFound",
        }
        response.status_code = 404
        with self.assertRaisesRegex(EntryNotFoundError, "Entry Not Found") as context:
            _raise_for_status(response)

        self.assertEqual(context.exception.response.status_code, 404)

    def test_raise_for_status_fallback(self):
        response = Response()
        response.status_code = 404
        response.headers = {
            "X-Request-Id": "test-id",
        }
        response.url = "test_URL"
        with self.assertRaisesRegex(HTTPError, "Request ID: test-id") as context:
            _raise_for_status(response)

        self.assertEqual(context.exception.response.status_code, 404)

    def test_raise_with_request_id(self):
        response = Response()
        response.status_code = 404
        response.headers = {
            "X-Request-Id": "test-id",
        }
        response.url = "test_URL"
        with self.assertRaisesRegex(HTTPError, "Request ID: test-id") as context:
            _raise_with_request_id(response)

        self.assertEqual(context.exception.response.status_code, 404)
