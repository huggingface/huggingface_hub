import inspect
import json
from typing import List, Optional, Union, get_args, get_origin

import pytest

import huggingface_hub.inference._generated.types as types
from huggingface_hub.inference._generated.types import AutomaticSpeechRecognitionParameters
from huggingface_hub.inference._generated.types.base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class DummyType(BaseInferenceType):
    foo: int
    bar: str


@dataclass_with_extra
class DummyNestedType(BaseInferenceType):
    item: DummyType
    items: List[DummyType]  # works both with List and list
    maybe_items: Optional[list[DummyType]] = None


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


def test_parse_from_unexpected_type():
    with pytest.raises(ValueError):
        DummyType.parse_obj(42)


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
    assert isinstance(instance, DummyNestedType)
    assert instance.item is None
    assert instance.items is None
    assert len(instance.maybe_items) == 2
    assert instance.maybe_items[0].foo is None
    assert instance.maybe_items[0].bar is None
    assert instance.maybe_items[1].foo == 42
    assert instance.maybe_items[1].bar == "baz"


def test_normalize_keys():
    # all fields are normalized in the dataclasses (by convention)
    # if server response uses different keys, they will be normalized
    instance = DummyNestedType.parse_obj({"ItEm": DUMMY_AS_DICT, "Maybe-Items": [DUMMY_AS_DICT]})
    assert isinstance(instance.item, DummyType)
    assert isinstance(instance.maybe_items, list)
    assert len(instance.maybe_items) == 1
    assert isinstance(instance.maybe_items[0], DummyType)


def test_optional_are_set_to_none():
    for _type in types.BaseInferenceType.__subclasses__():
        parameters = inspect.signature(_type).parameters
        for parameter in parameters.values():
            if _is_optional(parameter.annotation):
                assert parameter.default is None, f"Parameter {parameter} of {_type} should be set to None"


def test_none_inferred():
    """Regression test for https://github.com/huggingface/huggingface_hub/pull/2095"""
    # Doing this should not fail with
    # TypeError: __init__() missing 2 required positional arguments: 'generate' and 'return_timestamps'
    AutomaticSpeechRecognitionParameters()


def test_other_fields_are_set():
    instance = DummyNestedType.parse_obj(
        {
            "item": DUMMY_AS_DICT,
            "extra": "value",
            "items": [{"foo": 42, "another_extra": "value", "bar": "baz"}],
            "maybe_items": None,
        }
    )
    assert instance.extra == "value"
    assert instance.items[0].another_extra == "value"
    assert str(instance.items[0]) == "DummyType(foo=42, bar='baz', another_extra='value')"  # extra field always last
    assert (
        repr(instance)  # works both with __str__ and __repr__
        == (
            "DummyNestedType("
            "item=DummyType(foo=42, bar='baz'), "
            "items=[DummyType(foo=42, bar='baz', another_extra='value')], "
            "maybe_items=None, extra='value'"
            ")"
        )
    )


def test_other_fields_not_proper_dataclass_fields():
    instance_1 = DummyType.parse_obj({"foo": 42, "bar": "baz", "extra": "value1"})
    instance_2 = DummyType.parse_obj({"foo": 42, "bar": "baz", "extra": "value2", "another_extra": "value2.1"})
    assert instance_1.extra == "value1"
    assert instance_2.extra == "value2"
    assert instance_2.another_extra == "value2.1"

    # extra fields are not part of the dataclass fields
    # all dataclass methods except __repr__ should work as if the extra fields were not there
    assert instance_1 == instance_2


def _is_optional(field) -> bool:
    # Taken from https://stackoverflow.com/a/58841311
    return get_origin(field) is Union and type(None) in get_args(field)
