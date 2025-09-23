import json
import sys
from typing import Optional, Type, Union

import pytest

from huggingface_hub.utils._typing import is_jsonable, is_simple_optional_type, unwrap_simple_optional_type


class NotSerializableClass:
    pass


class CustomType:
    pass


OBJ_WITH_CIRCULAR_REF = {"hello": "world"}
OBJ_WITH_CIRCULAR_REF["recursive"] = OBJ_WITH_CIRCULAR_REF

_nested = {"hello": "world"}
OBJ_WITHOUT_CIRCULAR_REF = {"hello": _nested, "world": [_nested]}


@pytest.mark.parametrize(
    "data",
    [
        123,  #
        3.14,
        "Hello, world!",
        True,
        None,
        [],
        [1, 2, 3],
        [(1, 2.0, "string"), True],
        {},
        {"name": "Alice", "age": 30},
        {0: "LABEL_0", 1.0: "LABEL_1"},
        OBJ_WITHOUT_CIRCULAR_REF,
    ],
)
def test_is_jsonable_success(data):
    assert is_jsonable(data)
    json.dumps(data)


@pytest.mark.parametrize(
    "data",
    [
        set([1, 2, 3]),
        lambda x: x + 1,
        NotSerializableClass(),
        {"obj": NotSerializableClass()},
        OBJ_WITH_CIRCULAR_REF,
    ],
)
def test_is_jsonable_failure(data):
    assert not is_jsonable(data)
    with pytest.raises((TypeError, ValueError)):
        json.dumps(data)


@pytest.mark.parametrize(
    "type_, is_optional",
    [
        (Optional[int], True),
        (Union[None, int], True),
        (Union[int, None], True),
        (Optional[CustomType], True),
        (Union[None, CustomType], True),
        (Union[CustomType, None], True),
        (int, False),
        (None, False),
        (Union[int, float, None], False),
        (Union[Union[int, float], None], False),
        (Optional[Union[int, float]], False),
    ],
)
def test_is_simple_optional_type(type_: Type, is_optional: bool):
    assert is_simple_optional_type(type_) is is_optional


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
@pytest.mark.parametrize(
    "type_, is_optional",
    [
        ("int | None", True),
        ("None | int", True),
        ("CustomType | None", True),
        ("None | CustomType", True),
        ("int | float", False),
        ("int | float | None", False),
        ("(int | float) | None", False),
        ("Union[int, float] | None", False),
    ],
)
def test_is_simple_optional_type_pipe(type_: str, is_optional: bool):
    assert is_simple_optional_type(eval(type_)) is is_optional


@pytest.mark.parametrize(
    "optional_type, inner_type",
    [
        (Optional[int], int),
        (Union[int, None], int),
        (Union[None, int], int),
        (Optional[CustomType], CustomType),
        (Union[CustomType, None], CustomType),
        (Union[None, CustomType], CustomType),
    ],
)
def test_unwrap_simple_optional_type(optional_type: Type, inner_type: Type):
    assert unwrap_simple_optional_type(optional_type) is inner_type


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
@pytest.mark.parametrize(
    "optional_type, inner_type",
    [
        ("None | int", int),
        ("int | None", int),
        ("None | CustomType", CustomType),
        ("CustomType | None", CustomType),
    ],
)
def test_unwrap_simple_optional_type_pipe(optional_type: str, inner_type: Type):
    assert unwrap_simple_optional_type(eval(optional_type)) is inner_type


@pytest.mark.parametrize("non_optional_type", [int, None, CustomType])
def test_unwrap_simple_optional_type_fail(non_optional_type: Type):
    with pytest.raises(ValueError):
        unwrap_simple_optional_type(non_optional_type)
