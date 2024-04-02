"""Contains all custom errors."""

from requests import HTTPError


# INFERENCE CLIENT ERRORS


class InferenceTimeoutError(HTTPError, TimeoutError):
    """Error raised when a model is unavailable or the request times out."""


# TEXT GENERATION ERRORS


class TextGenerationError(HTTPError):
    """Generic error raised if text-generation went wrong."""


# Text Generation Inference Errors
class ValidationError(TextGenerationError):
    """Server-side validation error."""


class GenerationError(TextGenerationError):
    pass


class OverloadedError(TextGenerationError):
    pass


class IncompleteGenerationError(TextGenerationError):
    pass


class UnknownError(TextGenerationError):
    pass
