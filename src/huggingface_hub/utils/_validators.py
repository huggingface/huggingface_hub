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
"""Contains utilities to validate argument values in `huggingface_hub`."""
import inspect
import re
from functools import wraps
from itertools import chain
from typing import TypeVar


REPO_ID_REGEX = re.compile(
    r"""
    ^
    (\b[\w\-.]+\b/)? # optional namespace (username or organization)
    \b               # starts with a word boundary
    [\w\-.]{1,96}    # repo_name: alphanumeric + . _ -
    \b               # ends with a word boundary
    $
    """,
    flags=re.VERBOSE,
)


class HFValidationError(ValueError):
    """Generic exception thrown by `huggingface_hub` validators.

    Inherits from [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError).
    """


# type hint meaning "function signature not changed by decorator"
CallableT = TypeVar("CallableT")  # callable type


def validate_hf_hub_args(fn: CallableT) -> CallableT:
    """Validate values received as argument for any public method of `huggingface_hub`.

    The goal of this decorator is to harmonize validation of arguments reused
    everywhere. By default, all defined validators are tested.

    Validators:
        - [`~utils.validate_repo_id`]: `repo_id` must be `"repo_name"`
          or `"namespace/repo_name"`. Namespace is a username or an organization.

    Example:
    ```py
    >>> from huggingface_hub.utils import validate_hf_hub_args

    >>> @validate_hf_hub_args
    ... def my_cool_method(repo_id: str):
    ...     print(repo_id)

    >>> my_cool_method(repo_id="valid_repo_id")
    valid_repo_id

    >>> my_cool_method("other..repo..id")
    huggingface_hub.utils._validators.HFValidationError: Cannot have -- or .. in repo_id: 'other..repo..id'.

    >>> my_cool_method(repo_id="other..repo..id")
    huggingface_hub.utils._validators.HFValidationError: Cannot have -- or .. in repo_id: 'other..repo..id'.
    ```

    <Tip warning={true}>

    Raises:
        [`~utils.HFValidationError`]: If an input is not valid.

    </Tip>
    """
    # TODO: add an argument to opt-out validation for specific argument?
    signature = inspect.signature(fn)

    @wraps(fn)
    def _inner_fn(*args, **kwargs):
        for arg_name, arg_value in chain(
            zip(signature.parameters, args),  # Args values
            kwargs.items(),  # Kwargs values
        ):
            if arg_name == "repo_id":
                validate_repo_id(arg_value)

        return fn(*args, **kwargs)

    return _inner_fn


def validate_repo_id(repo_id: str) -> None:
    """Validate `repo_id` is valid.

    This is not meant to replace the proper validation made on the Hub but rather to
    avoid local inconsistencies whenever possible (example: passing `repo_type` in the
    `repo_id` is forbidden).

    Rules:
    - Between 1 and 96 characters.
    - Either "repo_name" or "namespace/repo_name"
    - [a-zA-Z0-9] or "-", "_", "."
    - "--" and ".." are forbidden

    Valid: `"foo"`, `"foo/bar"`, `"123"`, `"Foo-BAR_foo.bar123"`

    Not valid: `"datasets/foo/bar"`, `".repo_id"`, `"foo--bar"`, `"foo.git"`

    Example:
    ```py
    >>> from huggingface_hub.utils import validate_repo_id
    >>> validate_repo_id(repo_id="valid_repo_id")
    >>> validate_repo_id(repo_id="other..repo..id")
    huggingface_hub.utils._validators.HFValidationError: Cannot have -- or .. in repo_id: 'other..repo..id'.
    ```

    Discussed in https://github.com/huggingface/huggingface_hub/issues/1008.
    In moon-landing (internal repository):
    - https://github.com/huggingface/moon-landing/blob/main/server/lib/Names.ts#L27
    - https://github.com/huggingface/moon-landing/blob/main/server/views/components/NewRepoForm/NewRepoForm.svelte#L138
    """
    if not isinstance(repo_id, str):
        # Typically, a Path is not a repo_id
        raise HFValidationError(
            f"Repo id must be a string, not {type(repo_id)}: '{repo_id}'."
        )

    if repo_id.count("/") > 1:
        raise HFValidationError(
            "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"
            f" '{repo_id}'. Use `repo_type` argument if needed."
        )

    if not REPO_ID_REGEX.match(repo_id):
        raise HFValidationError(
            "Repo id must use alphanumeric chars or '-', '_', '.', '--' and '..' are"
            " forbidden, '-' and '.' cannot start or end the name, max length is 96:"
            f" '{repo_id}'."
        )

    if "--" in repo_id or ".." in repo_id:
        raise HFValidationError(f"Cannot have -- or .. in repo_id: '{repo_id}'.")

    if repo_id.endswith(".git"):
        raise HFValidationError(f"Repo_id cannot end by '.git': '{repo_id}'.")
