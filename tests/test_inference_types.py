import json
from dataclasses import dataclass

import pytest

from huggingface_hub.inference._generated.types import BaseInferenceType


@dataclass
class DummyType(BaseInferenceType):
    foo: int
    bar: str

DUMMY_AS_DICT = {"foo": 42, "bar": "baz"}
DUMMY_AS_STR = json.dumps(DUMMY_AS_DICT)
DUMMY_AS_BYTES = DUMMY_AS_STR.encode()
DUMMY_AS_LIST = [DUMMY_AS_DICT]

def test_parse_from_bytes():
    instance = DummyType.parse_obj(DUMMY_AS_BYTES)
    assert instance.foo == 42
    assert instance.bar == "baz"


def test_parse_from_str():
    instance = DummyType.parse_obj(DUMMY_AS_STR)
    assert instance.foo == 42
    assert instance.bar == "baz"


def test_parse_from_dict():
    instance = DummyType.parse_obj(DUMMY_AS_DICT)
    assert instance.foo == 42
    assert instance.bar == "baz"

def test_parse_from_list():
    instances = DummyType.parse_obj(DUMMY_AS_LIST)
    assert len(instances) == 1
    assert instances[0].foo == 42
    assert instances[0].bar == "baz"

def test_parse_as_list_success():
    instances = DummyType.parse_obj_as_list(DUMMY_AS_LIST)
    assert len(instances) == 1

def test_parse_as_list_failure():
    with pytest.raises(ValueError):
        DummyType.parse_obj_as_list(DUMMY_AS_DICT)