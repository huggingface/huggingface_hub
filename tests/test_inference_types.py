import json
from dataclasses import dataclass
from typing import List, Optional

import pytest

from huggingface_hub.inference._generated.types import BaseInferenceType

from .testing_utils import expect_deprecation


@dataclass
class DummyType(BaseInferenceType):
    foo: int
    bar: str


@dataclass
class DummyNestedType(BaseInferenceType):
    item: DummyType
    items: List[DummyType]
    maybe_items: Optional[List[DummyType]] = None

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


def test_parse_as_instance_success():
    instance = DummyType.parse_obj_as_instance(DUMMY_AS_DICT)
    assert isinstance(instance, DummyType)


def test_parse_as_instance_failure():
    with pytest.raises(ValueError):
        DummyType.parse_obj_as_instance(DUMMY_AS_LIST)


def test_parse_as_list_success():
    instances = DummyType.parse_obj_as_list(DUMMY_AS_LIST)
    assert len(instances) == 1


def test_parse_as_list_failure():
    with pytest.raises(ValueError):
        DummyType.parse_obj_as_list(DUMMY_AS_DICT)


def test_parse_nested_class():
    instance = DummyNestedType.parse_obj(
        {
            "item": DUMMY_AS_DICT,
            "items": DUMMY_AS_LIST,
            "maybe_items": None,
        }
    )
    assert instance.item.foo == 42
    assert instance.item.bar == "baz"
    assert len(instance.items) == 1
    assert instance.items[0].foo == 42
    assert instance.items[0].bar == "baz"
    assert instance.maybe_items is None


def test_all_fields_are_optional():
    # all fields are optional => silently accept None if server returns less data than expected
    instance = DummyNestedType.parse_obj({"maybe_items": [{}, DUMMY_AS_BYTES]})
    assert instance.item is None
    assert instance.items is None
    assert len(instance.maybe_items) == 2
    assert instance.maybe_items[0].foo is None
    assert instance.maybe_items[0].bar is None
    assert instance.maybe_items[1].foo == 42
    assert instance.maybe_items[1].bar == "baz"


@expect_deprecation("DummyType")
def test_not_expected_fields():
    instance = DummyType.parse_obj({"foo": 42, "bar": "baz", "not_expected": "value"})
    assert instance.foo == 42
    assert instance.bar == "baz"
    assert instance["not_expected"] == "value"


@expect_deprecation("DummyType")
def test_fields_kept_in_sync():
    # hacky but works + will be removed once dataclasses will not be dicts anymore
    instance = DummyType.parse_obj(DUMMY_AS_DICT)
    assert instance.foo == 42
    assert instance["foo"] == 42

    instance.foo = 43
    assert instance.foo == 43
    assert instance["foo"] == 43

    instance["foo"] = 44
    assert instance.foo == 44
    assert instance["foo"] == 44

def test_normalize_keys():
    # all fields are normalized in the dataclasses (by convention)
    # if server response uses different keys, they will be normalized
    instance = DummyNestedType.parse_obj({"ItEm": DUMMY_AS_DICT, "Maybe-Items": [DUMMY_AS_DICT]})
    assert isinstance(instance.item, DummyType)
    assert isinstance(instance.maybe_items, list)
    assert len(instance.maybe_items) == 1
    assert isinstance(instance.maybe_items[0], DummyType)