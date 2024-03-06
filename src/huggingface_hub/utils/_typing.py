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
"""Handle typing imports based on system compatibility."""

from typing import Any, Callable, Literal, TypeVar


HTTP_METHOD_T = Literal["GET", "OPTIONS", "HEAD", "POST", "PUT", "PATCH", "DELETE"]

# type hint meaning "function signature not changed by decorator"
CallableT = TypeVar("CallableT", bound=Callable)

_JSON_SERIALIZABLE_TYPES = (int, float, str, bool, type(None))


def is_jsonable(obj: Any) -> bool:
    """Check if an object is JSON serializable.

    This is a weak check, as it does not check for the actual JSON serialization, but only for the types of the object.
    It works correctly for basic use cases but do not guarantee an exhaustive check.

    Object is considered to be recursively json serializable if:
    - it is an instance of int, float, str, bool, or NoneType
    - it is a list or tuple and all its items are json serializable
    - it is a dict and all its keys are strings and all its values are json serializable
    """
    try:
        if isinstance(obj, _JSON_SERIALIZABLE_TYPES):
            return True
        if isinstance(obj, (list, tuple)):
            return all(is_jsonable(item) for item in obj)
        if isinstance(obj, dict):
            return all(isinstance(key, str) and is_jsonable(value) for key, value in obj.items())
        if hasattr(obj, "__json__"):
            return True
        return False
    except RecursionError:
        return False
