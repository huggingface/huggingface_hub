"""Contains all custom errors."""

from requests import HTTPError


class InferenceTimeoutError(HTTPError, TimeoutError):
    """Error raised when a model is unavailable or the request times out."""
