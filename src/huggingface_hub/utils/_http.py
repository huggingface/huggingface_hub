# coding=utf-8
# Copyright 2022-present, the HuggingFace Inc. team.
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
"""Contains utilities to handle HTTP requests in Huggingface Hub."""
import io
import time
from http import HTTPStatus
from typing import Tuple, Type, Union

import requests
from requests import Response
from requests.exceptions import ConnectTimeout, ProxyError

from . import logging
from ._typing import HTTP_METHOD_T


logger = logging.get_logger(__name__)


def http_backoff(
    method: HTTP_METHOD_T,
    url: str,
    *,
    max_retries: int = 5,
    base_wait_time: float = 1,
    max_wait_time: float = 8,
    retry_on_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = (
        ConnectTimeout,
        ProxyError,
    ),
    retry_on_status_codes: Union[int, Tuple[int, ...]] = HTTPStatus.SERVICE_UNAVAILABLE,
    **kwargs,
) -> Response:
    """Wrapper around requests to retry calls on an endpoint, with exponential backoff.

    Endpoint call is retried on exceptions (ex: connection timeout, proxy error,...)
    and/or on specific status codes (ex: service unavailable). If the call failed more
    than `max_retries`, the exception is thrown or `raise_for_status` is called on the
    response object.

    Re-implement mechanisms from the `backoff` library to avoid adding an external
    dependencies to `hugging_face_hub`. See https://github.com/litl/backoff.

    Args:
        method (`Literal["GET", "OPTIONS", "HEAD", "POST", "PUT", "PATCH", "DELETE"]`):
            HTTP method to perform.
        url (`str`):
            The URL of the resource to fetch.
        max_retries (`int`, *optional*, defaults to `5`):
            Maximum number of retries, defaults to 5 (no retries).
        base_wait_time (`float`, *optional*, defaults to `1`):
            Duration (in seconds) to wait before retrying the first time.
            Wait time between retries then grows exponentially, capped by
            `max_wait_time`.
        max_wait_time (`float`, *optional*, defaults to `8`):
            Maximum duration (in seconds) to wait before retrying.
        retry_on_exceptions (`Type[Exception]` or `Tuple[Type[Exception]]`, *optional*, defaults to `(ConnectTimeout, ProxyError,)`):
            Define which exceptions must be caught to retry the request. Can be a single
            type or a tuple of types.
            By default, retry on `ConnectTimeout` and `ProxyError`.
        retry_on_status_codes (`int` or `Tuple[int]`, *optional*, defaults to `503`):
            Define on which status codes the request must be retried. By default, only
            HTTP 503 Service Unavailable is retried.
        **kwargs (`dict`, *optional*):
            kwargs to pass to `requests.request`.

    Example:
    ```
    >>> from huggingface_hub.utils import http_backoff

    # Same usage as "requests.request".
    >>> response = http_backoff("GET", "https://www.google.com")
    >>> response.raise_for_status()

    # If you expect a Gateway Timeout from time to time
    >>> http_backoff("PUT", upload_url, data=data, retry_on_status_codes=504)
    >>> response.raise_for_status()
    ```

    <Tip warning={true}>

    When using `requests` it is possible to stream data by passing an iterator to the
    `data` argument. On http backoff this is a problem as the iterator is not reset
    after a failed call. This issue is mitigated for file objects or any IO streams
    by saving the initial position of the cursor (with `data.tell()`) and resetting the
    cursor between each call (with `data.seek()`). For arbitrary iterators, http backoff
    will fail. If this is a hard constraint for you, please let us know by opening an
    issue on [Github](https://github.com/huggingface/huggingface_hub).

    </Tip>
    """
    if isinstance(retry_on_exceptions, type):  # Tuple from single exception type
        retry_on_exceptions = (retry_on_exceptions,)

    if isinstance(retry_on_status_codes, int):  # Tuple from single status code
        retry_on_status_codes = (retry_on_status_codes,)

    nb_tries = 0
    sleep_time = base_wait_time

    # If `data` is used and is a file object (or any IO), it will be consumed on the
    # first HTTP request. We need to save the initial position so that the full content
    # of the file is re-sent on http backoff. See warning tip in docstring.
    io_obj_initial_pos = None
    if "data" in kwargs and isinstance(kwargs["data"], io.IOBase):
        io_obj_initial_pos = kwargs["data"].tell()

    while True:
        nb_tries += 1
        try:
            # If `data` is used and is a file object (or any IO), set back cursor to
            # initial position.
            if io_obj_initial_pos is not None:
                kwargs["data"].seek(io_obj_initial_pos)

            # Perform request and return if status_code is not in the retry list.
            response = requests.request(method=method, url=url, **kwargs)
            if response.status_code not in retry_on_status_codes:
                return response

            # Wrong status code returned (HTTP 503 for instance)
            logger.warning(
                f"HTTP Error {response.status_code} thrown while requesting"
                f" {method} {url}"
            )
            if nb_tries > max_retries:
                response.raise_for_status()  # Will raise uncaught exception
                # We return response to avoid infinite loop in the corner case where the
                # user ask for retry on a status code that doesn't raise_for_status.
                return response

        except retry_on_exceptions as err:
            logger.warning(f"'{err}' thrown while requesting {method} {url}")

            if nb_tries > max_retries:
                raise err

        # Sleep for X seconds
        logger.warning(f"Retrying in {sleep_time}s [Retry {nb_tries}/{max_retries}].")
        time.sleep(sleep_time)

        # Update sleep time for next retry
        sleep_time = min(max_wait_time, sleep_time * 2)  # Exponential backoff
