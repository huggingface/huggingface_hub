import json

import pytest

from huggingface_hub.utils._typing import is_jsonable


class NotSerializableClass:
    pass


OBJ_WITH_CIRCULAR_REF = {"hello": "world"}
OBJ_WITH_CIRCULAR_REF["recursive"] = OBJ_WITH_CIRCULAR_REF


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
