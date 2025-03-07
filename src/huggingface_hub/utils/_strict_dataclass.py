import inspect
from dataclasses import Field, dataclass, field, fields
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union, get_args, get_origin

from ..errors import StrictDataclassDefinitionError, StrictDataclassFieldValidationError


Validator_T = Callable[[Any], None]
T = TypeVar("T")


def validated_field(validator: Union[List[Validator_T], Validator_T], **kwargs: Any) -> Any:
    """
    Create a dataclass field with a custom validator.

    Args:
        validator (`Callable` or `List[Callable]`):
            A method that takes a value as input and raises ValueError/TypeError if the value is invalid.
            Can be a list of validators to apply multiple checks.
        **kwargs:
            Additional arguments to pass to `dataclasses.field()`.

    Returns:
        A field with the validator attached in metadata
    """
    metadata = kwargs.pop("metadata", {})
    if not isinstance(validator, list):
        validator = [validator]
    metadata["validator"] = validator
    return field(metadata=metadata, **kwargs)


def strict_dataclass(cls: Type[T]) -> Type[T]:
    """
    Decorator to create a strict dataclass with type validation.

    Args:
        cls: The class to convert to a strict dataclass.

    Returns:
        A dataclass with strict type validation on field assignment.

    Raises:
        [`StrictDataclassDefinitionError`]: If the class definition is invalid

    Example:
    ```python
    >>> from huggingface_hub.utils import strict_dataclass, validated_field

    >>> def positive_int(value: int):
    ...     if not value >= 0:
    ...         raise ValueError(f"Value must be positive, got {value}")

    >>> @strict_dataclass
    ... class User:
    ...     name: str
    ...     age: int = validated_field(positive_int)
    >>> user = User(name="John", age=30)

    >>> User(name="John", age="30")  # Invalid type
    StrictDataclassFieldValidationError(...)
    ```
    """
    cls = dataclass(cls)

    # List all validator fields
    field_validators: Dict[str, List[Validator_T]] = {}
    for f in fields(cls):  # type: ignore[arg-type]
        validators = []
        validators.append(_create_type_validator(f))
        if custom_validator := f.metadata.get("validator"):
            if not isinstance(custom_validator, list):
                custom_validator = [custom_validator]
            for validator in custom_validator:
                if not _is_validator(validator):
                    raise StrictDataclassDefinitionError(
                        f"Invalid validator for field '{f.name}': {validator}. Must be a callable taking a single argument."
                    )
            validators.extend(custom_validator)
        field_validators[f.name] = validators

    cls.__validators__ = field_validators  # type: ignore

    # Override __setattr__ to type_validator on assignment
    cls.__setattr__ = _strict_setattr  # type: ignore[method-assign]

    return cls


def type_validator(name: str, value: Any, expected_type: Any) -> None:
    """Validate that 'value' matches 'expected_type'."""
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if expected_type is Any:
        return
    elif origin is Union:
        _validate_union(name, value, args)
    elif origin is Optional:
        _validate_optional(name, value, args)
    elif origin is Literal:
        _validate_literal(name, value, args)
    elif origin is list:
        _validate_list(name, value, args)
    elif origin is dict:
        _validate_dict(name, value, args)
    elif origin is tuple:
        _validate_tuple(name, value, args)
    elif origin is set:
        _validate_set(name, value, args)
    elif isinstance(expected_type, type):
        _validate_simple_type(name, value, expected_type)
    else:
        raise TypeError(f"Unsupported type for field '{name}': {expected_type}")


def _validate_union(name: str, value: Any, args: Tuple[Any, ...]) -> None:
    """Validate that value matches one of the types in a Union."""
    errors = []
    for t in args:
        try:
            type_validator(name, value, t)
            return  # Valid if any type matches
        except TypeError as e:
            errors.append(str(e))

    raise TypeError(
        f"Field '{name}' with value {repr(value)} doesn't match any type in {args}. Errors: {'; '.join(errors)}"
    )


def _validate_optional(name: str, value: Any, args: Tuple[Any, ...]) -> None:
    """Validate Optional[T] type."""
    if value is not None:
        type_validator(name, value, args[0])


def _validate_literal(name: str, value: Any, args: Tuple[Any, ...]) -> None:
    """Validate Literal type."""
    if value not in args:
        raise TypeError(f"Field '{name}' expected one of {args}, got {value}")


def _validate_list(name: str, value: Any, args: Tuple[Any, ...]) -> None:
    """Validate List[T] type."""
    if not isinstance(value, list):
        raise TypeError(f"Field '{name}' expected a list, got {type(value).__name__}")

    # Validate each item in the list
    item_type = args[0]
    for i, item in enumerate(value):
        try:
            type_validator(f"{name}[{i}]", item, item_type)
        except TypeError as e:
            raise TypeError(f"Invalid item at index {i} in list '{name}'") from e


def _validate_dict(name: str, value: Any, args: Tuple[Any, ...]) -> None:
    """Validate Dict[K, V] type."""
    if not isinstance(value, dict):
        raise TypeError(f"Field '{name}' expected a dict, got {type(value).__name__}")

    # Validate keys and values
    key_type, value_type = args
    for k, v in value.items():
        try:
            type_validator(f"{name}.key", k, key_type)
            type_validator(f"{name}[{k!r}]", v, value_type)
        except TypeError as e:
            raise TypeError(f"Invalid key or value in dict '{name}'") from e


def _validate_tuple(name: str, value: Any, args: Tuple[Any, ...]) -> None:
    """Validate Tuple type."""
    if not isinstance(value, tuple):
        raise TypeError(f"Field '{name}' expected a tuple, got {type(value).__name__}")

    # Handle variable-length tuples: Tuple[T, ...]
    if len(args) == 2 and args[1] is Ellipsis:
        for i, item in enumerate(value):
            try:
                type_validator(f"{name}[{i}]", item, args[0])
            except TypeError as e:
                raise TypeError(f"Invalid item at index {i} in tuple '{name}'") from e
    # Handle fixed-length tuples: Tuple[T1, T2, ...]
    elif len(args) != len(value):
        raise TypeError(f"Field '{name}' expected a tuple of length {len(args)}, got {len(value)}")
    else:
        for i, (item, expected) in enumerate(zip(value, args)):
            try:
                type_validator(f"{name}[{i}]", item, expected)
            except TypeError as e:
                raise TypeError(f"Invalid item at index {i} in tuple '{name}'") from e


def _validate_set(name: str, value: Any, args: Tuple[Any, ...]) -> None:
    """Validate Set[T] type."""
    if not isinstance(value, set):
        raise TypeError(f"Field '{name}' expected a set, got {type(value).__name__}")

    # Validate each item in the set
    item_type = args[0]
    for i, item in enumerate(value):
        try:
            type_validator(f"{name} item", item, item_type)
        except TypeError as e:
            raise TypeError(f"Invalid item in set '{name}'") from e


def _validate_simple_type(name: str, value: Any, expected_type: type) -> None:
    """Validate simple type (int, str, etc.)."""
    if not isinstance(value, expected_type):
        raise TypeError(
            f"Field '{name}' expected {expected_type.__name__}, got {type(value).__name__} (value: {repr(value)})"
        )


def _create_type_validator(field: Field) -> Validator_T:
    """Create a type validator function for a field."""
    # Hacky: we cannot use a lambda here because of reference issues

    def validator(value: Any) -> None:
        type_validator(field.name, value, field.type)

    return validator


def _strict_setattr(self: Any, name: str, value: Any) -> None:
    """Custom __setattr__ method for strict dataclasses.

    Check is lax on new attributes, but strict on existing ones.
    """
    # Run all validators
    for validator in self.__validators__.get(name, []):
        try:
            validator(value)
        except (ValueError, TypeError) as e:
            raise StrictDataclassFieldValidationError(field=name, cause=e) from e

    # If validation passed, set the attribute
    super(self.__class__, self).__setattr__(name, value)


def _is_validator(validator: Any) -> bool:
    """Check if a function is a validator.

    A validator is a Callable that can be called with a single positional argument.
    The validator can have more arguments with default values.

    Basically, returns True if `validator(value)` is possible.
    """
    if not callable(validator):
        return False

    signature = inspect.signature(validator)
    parameters = list(signature.parameters.values())
    if len(parameters) == 0:
        return False
    if parameters[0].kind not in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.VAR_POSITIONAL,
    ):
        return False
    for parameter in parameters[1:]:
        if parameter.default == inspect.Parameter.empty:
            return False
    return True
