import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Type, TypeVar, Union, get_args, get_origin


T = TypeVar("T", bound="BaseInferenceType")


@dataclass
class BaseInferenceType(dict):
    """Base class for all inference types."""

    @classmethod
    def from_data(cls: Type[T], data: Union[bytes, str, List, Dict]) -> Union[List[T], T]:
        """Parse server response as a dataclass.

        To enable future-compatibility, we want to handle cases where the server return more fields than expected.
        In such cases, we don't want to raise an error but still create the dataclass object. Remaining fields are
        added as dict attributes.
        """
        # Parse server response (from bytes)
        if isinstance(data, bytes):
            data = data.decode()
        if isinstance(data, str):
            data = json.loads(data)

        # If a list, parse each item individually
        if isinstance(data, List):
            return [cls.from_data(d) for d in data]  # type: ignore [misc]

        # At this point, we expect a dict
        if not isinstance(data, dict):
            raise ValueError(f"Invalid data type: {type(data)}")

        init_values = {}
        other_values = {}
        for key, value in data.items():
            if key in cls.__dataclass_fields__ and cls.__dataclass_fields__[key].init:
                if isinstance(value, dict) or isinstance(value, list):
                    # Recursively parse nested dataclasses (if possible)
                    # `get_args` returns handle Union and Optional for us
                    expected_types = get_args(cls.__dataclass_fields__[key].type)
                    for expected_type in expected_types:
                        if getattr(expected_type, "_name", None) == "List":
                            expected_type = get_args(expected_type)[0]  # assume same type for all items in the list
                        if issubclass(expected_type, BaseInferenceType):
                            if isinstance(value, dict):
                                value = expected_type.from_data(value)
                            elif isinstance(value, list):
                                value = [expected_type.from_data(v) for v in value]
                            break
                init_values[key] = value
            else:
                other_values[key] = value

        # Make all optional fields default to None
        for key, field in cls.__dataclass_fields__.items():
            if key not in init_values:
                if is_optional(field.type):
                    init_values[key] = None

        # Initialize dataclass with expected values
        item = cls(**init_values)

        # Add remaining fields as dict attributes
        item.update(other_values)
        return item

    def __post_init__(self):
        self.update(asdict(self))


def is_optional(field):
    # Check if a field is Optional
    # Taken from https://stackoverflow.com/a/58841311
    return get_origin(field) is Union and type(None) in get_args(field)
