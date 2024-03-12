# coding=utf-8
# Copyright 2023-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Original implementation taken from the `text-generation` Python client (see https://pypi.org/project/text-generation/
# and https://github.com/huggingface/text-generation-inference/tree/main/clients/python)
#
# Changes compared to original implementation:
# - use Python's dataclasses (e.g. no validation on output)
# - added default values for all parameters (not needed in BaseModel but dataclasses yes)
# - integrated in `huggingface_hub.InferenceClient`
# - added `stream: bool` and `details: bool` in the `text_generation` method instead of having different methods for each use case
from typing import NoReturn, Optional

from requests import HTTPError


# TEXT GENERATION ERRORS
# ----------------------
# Text-generation errors are parsed separately to handle as much as possible the errors returned by the text generation
# inference project (https://github.com/huggingface/text-generation-inference).
# ----------------------


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


def raise_text_generation_error(http_error: HTTPError) -> NoReturn:
    """
    Try to parse text-generation-inference error message and raise HTTPError in any case.

    Args:
        error (`HTTPError`):
            The HTTPError that have been raised.
    """
    # Try to parse a Text Generation Inference error

    try:
        # Hacky way to retrieve payload in case of aiohttp error
        payload = getattr(http_error, "response_error_payload", None) or http_error.response.json()
        error = payload.get("error")
        error_type = payload.get("error_type")
    except Exception:  # no payload
        raise http_error

    # If error_type => more information than `hf_raise_for_status`
    if error_type is not None:
        exception = _parse_text_generation_error(error, error_type)
        raise exception from http_error

    # Otherwise, fallback to default error
    raise http_error


def _parse_text_generation_error(error: Optional[str], error_type: Optional[str]) -> TextGenerationError:
    if error_type == "generation":
        return GenerationError(error)  # type: ignore
    if error_type == "incomplete_generation":
        return IncompleteGenerationError(error)  # type: ignore
    if error_type == "overloaded":
        return OverloadedError(error)  # type: ignore
    if error_type == "validation":
        return ValidationError(error)  # type: ignore
    return UnknownError(error)  # type: ignore
