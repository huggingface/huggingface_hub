import inspect
import sys
from dataclasses import asdict, astuple, dataclass, is_dataclass
from typing import Annotated, Any, Literal, Optional, TypedDict, Union, get_type_hints

import jedi
import pytest


if sys.version_info >= (3, 11):
    from typing import NotRequired, Required
else:
    # Provide fallbacks or skip the entire module
    NotRequired = None
    Required = None
from huggingface_hub.dataclasses import (
    _build_strict_cls_from_typed_dict,
    _is_validator,
    as_validated_field,
    strict,
    type_validator,
    validate_typed_dict,
    validated_field,
)
from huggingface_hub.errors import (
    StrictDataclassClassValidationError,
    StrictDataclassDefinitionError,
    StrictDataclassFieldValidationError,
)


def positive_int(value: int):
    if not value >= 0:
        raise ValueError(f"Value must be positive, got {value}")


def multiple_of_64(value: int):
    if not value % 64 == 0:
        raise ValueError(f"Value must be a multiple of 64, got {value}")


@as_validated_field
def strictly_positive(value: int):
    if not value > 0:
        raise ValueError(f"Value must be strictly positive, got {value}")


def dtype_validation(value: "ForwardDtype"):
    if not isinstance(value, str):
        raise ValueError(f"Value must be string, got {value}")

    if isinstance(value, str) and value not in ["float32", "bfloat16", "float16"]:
        raise ValueError(f"Value must be one of `[float32, bfloat16, float16]` but got {value}")


@strict
@dataclass
class ConfigForwardRef:
    """Test forward reference handling.

    In practice, forward reference types are not validated so a custom validator is highly recommended.
    """

    forward_ref_validated: "ForwardDtype" = validated_field(validator=dtype_validation)
    forward_ref: "ForwardDtype" = "float32"  # type is not validated by default


class ForwardDtype(str):
    """Dummy class to simulate a forward reference (e.g. `torch.dtype`)."""


@strict
@dataclass
class Config:
    model_type: str
    hidden_size: int = validated_field(validator=[positive_int, multiple_of_64])
    vocab_size: int = strictly_positive(default=16)


@strict(accept_kwargs=True)
@dataclass
class ConfigWithKwargs:
    model_type: str
    vocab_size: int = validated_field(validator=positive_int, default=16)


class DummyClass:
    pass


def test_valid_initialization():
    config = Config(model_type="bert", vocab_size=30000, hidden_size=768)
    assert config.model_type == "bert"
    assert config.vocab_size == 30000
    assert config.hidden_size == 768


def test_default_values():
    config = Config(model_type="bert", hidden_size=1024)
    assert config.model_type == "bert"
    assert config.vocab_size == 16
    assert config.hidden_size == 1024


def test_forward_ref_validation_is_skipped():
    config = ConfigForwardRef(forward_ref="float32", forward_ref_validated="float32")
    assert config.forward_ref == "float32"
    assert config.forward_ref_validated == "float32"

    # The `forward_ref_validated` has proper validation added in field-metadata and will be validated
    with pytest.raises(StrictDataclassFieldValidationError):
        ConfigForwardRef(forward_ref_validated="float64")

    with pytest.raises(StrictDataclassFieldValidationError):
        ConfigForwardRef(forward_ref_validated=-1)

    with pytest.raises(StrictDataclassFieldValidationError):
        ConfigForwardRef(forward_ref_validated="not_dtype")

    # The `forward_ref` type is not validated => user can input anything
    ConfigForwardRef(forward_ref=-1, forward_ref_validated="float32")
    ConfigForwardRef(forward_ref=["float32"], forward_ref_validated="float32")


def test_invalid_type_initialization():
    with pytest.raises(StrictDataclassFieldValidationError):
        Config(model_type={"type": "bert"}, vocab_size=30000, hidden_size=768)

    with pytest.raises(StrictDataclassFieldValidationError):
        Config(model_type="bert", vocab_size="30000", hidden_size=768)


def test_all_validators_are_applied():
    # must be positive
    with pytest.raises(StrictDataclassFieldValidationError):
        Config(model_type="bert", vocab_size=-1, hidden_size=1024)

    # must be a multiple of 64
    with pytest.raises(StrictDataclassFieldValidationError):
        Config(model_type="bert", hidden_size=1025)

    # both validators are applied
    with pytest.raises(StrictDataclassFieldValidationError):
        Config(model_type="bert", hidden_size=-1024)


def test_validated_on_assignment():
    config = Config(model_type="bert", hidden_size=1024)
    config.vocab_size = 10000  # ok
    with pytest.raises(StrictDataclassFieldValidationError):
        config.vocab_size = "10000"  # type validator checked
    with pytest.raises(StrictDataclassFieldValidationError):
        config.vocab_size = -1  # custom validators checked
    with pytest.raises(StrictDataclassFieldValidationError):
        config.vocab_size = 0  # must be strictly positive


def test_lax_on_new_attributes():
    config = Config(model_type="bert", hidden_size=1024)
    config.new_attribute = "new_value"
    assert config.new_attribute == "new_value"  # not validated


def test_custom_validator_must_be_callable():
    """Must raise at class definition time."""
    with pytest.raises(StrictDataclassDefinitionError):

        @strict
        @dataclass
        class Config:
            model_type: str = validated_field(validator="not_a_function")

    with pytest.raises(StrictDataclassDefinitionError):

        @strict
        @dataclass
        class Config:
            model_type: str = validated_field(validator=lambda: None)  # not a validator either


@pytest.mark.parametrize(
    "value, type_annotation",
    [
        # Basic types
        (5, int),
        (5.0, float),
        ("John", str),
        # Union types (typing.Union)
        (5, Union[int, str]),
        ("John", Union[int, str]),
        # Optional
        (5, Optional[int]),
        (None, Optional[int]),
        (DummyClass(), Optional[DummyClass]),
        # Literal
        ("John", Literal["John", "Doe"]),
        (5, Literal[4, 5, 6]),
        # List
        ([1, 2, 3], list[int]),
        ([1, 2, "3"], list[Union[int, str]]),
        # Tuple
        ((1, 2, 3), tuple[int, int, int]),
        ((1, 2, "3"), tuple[int, int, str]),
        ((1, 2, 3, 4), tuple[int, ...]),
        # Dict
        ({"a": 1, "b": 2}, dict[str, int]),
        ({"a": 1, "b": "2"}, dict[str, Union[int, str]]),
        # Set
        ({1, 2, 3}, set[int]),
        ({1, 2, "3"}, set[Union[int, str]]),
        # Custom classes
        (DummyClass(), DummyClass),
        # Any
        (5, Any),
        ("John", Any),
        (DummyClass(), Any),
        # Deep nested type
        (
            {
                "a": [
                    (1, DummyClass(), {1, "2", "3", 4}),
                    (2, DummyClass(), None),
                ],
            },
            dict[
                str,
                list[
                    tuple[
                        int,
                        DummyClass,
                        Optional[set[Union[int, str],]],
                    ]
                ],
            ],
        ),
    ],
)
def test_type_validator_valid(value, type_annotation):
    type_validator("dummy", value, type_annotation)


@pytest.mark.parametrize(
    "value, type_annotation",
    [
        # Basic types
        (5, float),
        (5.0, int),
        ("John", int),
        # Union types (typing.Union)
        (5.0, Union[int, str]),
        (None, Union[int, str]),
        (DummyClass(), Union[int, str]),
        # Optional
        ("John", Optional[int]),
        (DummyClass(), Optional[int]),
        # Literal
        ("Ada", Literal["John", "Doe"]),
        (3, Literal[4, 5, 6]),
        # List
        (5, list[int]),
        ([1, 2, "3"], list[int]),
        # Tuple
        (5, tuple[int, int, int]),
        ((1, 2, "3"), tuple[int, int, int]),
        ((1, 2, 3, 4), tuple[int, int, int]),
        ((1, 2, "3", 4), tuple[int, ...]),
        # Dict
        (5, dict[str, int]),
        ({"a": 1, "b": "2"}, dict[str, int]),
        # Set
        (5, set[int]),
        ({1, 2, "3"}, set[int]),
        # Custom classes
        (5, DummyClass),
        ("John", DummyClass),
    ],
)
def test_type_validator_invalid(value, type_annotation):
    with pytest.raises(TypeError):
        type_validator("dummy", value, type_annotation)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires Python 3.10+")
def test_type_union_type():
    # TODO: make it first class citizen when bumping to Python 3.10+
    # Union types (x | y syntax)
    for value, type_annotation in [
        (5, int | str),
        ("John", int | str),
        (None, int | None),
        (DummyClass(), DummyClass | int | None),
    ]:
        type_validator("dummy", value, type_annotation)

    for value, type_annotation in [
        (5.0, int | str),
        (None, int | str),
        (DummyClass(), int | str),
        ("str", DummyClass | int | None),
    ]:
        with pytest.raises(TypeError):
            type_validator("dummy", value, type_annotation)


class DummyValidator:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, value):
        return value < self.threshold

    def compare(self, value, value2=10):
        return value < value2


@pytest.mark.parametrize(
    "obj",
    [
        positive_int,
        multiple_of_64,
        lambda value: None,
        lambda value, factor=2: None,
        lambda value=1, factor=2: value * factor,
        lambda *values: None,
        DummyValidator(threshold=10),  # callable object
        DummyValidator(threshold=10).compare,  # callable method
    ],
)
def test_is_validator(obj):
    # Anything that can be called with `obj(value)` is a correct validator.
    assert _is_validator(obj)


@pytest.mark.parametrize(
    "obj",
    [
        5,  # not callable
        lambda: None,  # no argument
        lambda value1, value2: None,  # more than one argument with default values
        lambda *, value: None,  # keyword-only argument
    ],
)
def test_not_a_validator(obj):
    assert not _is_validator(obj)


def test_preserve_metadata():
    class ConfigWithMetadataField:
        foo: int = strictly_positive(metadata={"foo": "bar"}, default=10)

    assert ConfigWithMetadataField.foo.metadata["foo"] == "bar"


def test_accept_kwargs():
    config = ConfigWithKwargs(model_type="bert", vocab_size=30000, hidden_size=768)
    assert config.model_type == "bert"
    assert config.vocab_size == 30000
    assert config.hidden_size == 768

    # Defined fields are still validated
    with pytest.raises(StrictDataclassFieldValidationError):
        ConfigWithKwargs(model_type="bert", vocab_size=-1)

    # Default values are still used
    config = ConfigWithKwargs(model_type="bert")
    assert config.vocab_size == 16


def test_do_not_accept_kwargs():
    @strict
    @dataclass
    class Config:
        model_type: str

    with pytest.raises(TypeError):
        Config(model_type="bert", vocab_size=30000)


def test_is_recognized_as_dataclass():
    # Check that dataclasses module recognizes it as a dataclass
    assert is_dataclass(Config)

    # Check that an instance is recognized as a dataclass instance
    config = Config(model_type="bert", hidden_size=768)
    assert is_dataclass(config)


def test_behave_as_a_dataclass():
    # Check that dataclasses.asdict works
    config = Config(model_type="bert", hidden_size=768)
    assert asdict(config) == {"model_type": "bert", "hidden_size": 768, "vocab_size": 16}

    # Check that dataclasses.astuple works
    assert astuple(config) == ("bert", 768, 16)


def test_type_annotations_preserved():
    # Check that type hints are preserved
    hints = get_type_hints(Config)
    assert hints["model_type"] is str
    assert hints["hidden_size"] is int
    assert hints["vocab_size"] is int


def test_correct_init_signature():
    # Check that __init__ has the expected signature
    signature = inspect.signature(Config.__init__)
    parameters = list(signature.parameters.values())

    # First param should be self
    assert parameters[0].name == "self"

    # model_type should be required
    assert parameters[1].name == "model_type"
    assert parameters[1].default == inspect.Parameter.empty

    # hidden_size should be required (and validated)
    assert parameters[2].name == "hidden_size"
    assert parameters[2].default == inspect.Parameter.empty

    # vocab_size should be optional with default
    assert parameters[3].name == "vocab_size"
    assert parameters[3].default == 16


def test_correct_eq_repr():
    # Test equality comparison
    config1 = Config(model_type="bert", hidden_size=0)
    config2 = Config(model_type="bert", hidden_size=0)
    config3 = Config(model_type="gpt", hidden_size=0)

    assert config1 == config2
    assert config1 != config3

    # Test repr
    assert repr(config1) == "Config(model_type='bert', hidden_size=0, vocab_size=16)"


def test_repr_if_accept_kwargs():
    config1 = ConfigWithKwargs(foo="bar", model_type="bert")
    assert repr(config1) == "ConfigWithKwargs(model_type='bert', vocab_size=16, *foo='bar')"


def test_autocompletion_attribute_without_kwargs():
    # Create a sample script
    completions = jedi.Script("""
from dataclasses import dataclass
from huggingface_hub.dataclasses import strict

@strict
@dataclass
class Config:
    model_type: str
    hidden_size: int = 768

config = Config(model_type="bert")
config.
""").complete(line=12, column=7)
    completion_names = [c.name for c in completions]
    assert "model_type" in completion_names
    assert "hidden_size" in completion_names


def test_autocompletion_attribute_with_kwargs():
    # Create a sample script
    completions = jedi.Script("""
from dataclasses import dataclass
from huggingface_hub.dataclasses import strict

@strict(accept_kwargs=True)
@dataclass
class Config:
    model_type: str
    hidden_size: int = 768

config = Config(model_type="bert", foo="bar")
config.
""").complete(line=12, column=7)
    completion_names = [c.name for c in completions]
    assert "model_type" in completion_names
    assert "hidden_size" in completion_names
    assert "foo" not in completion_names  # not an official arg


def test_autocompletion_init_without_kwargs():
    # Create a sample script
    completions = jedi.Script("""
from dataclasses import dataclass
from huggingface_hub.dataclasses import strict

@strict
@dataclass
class Config:
    model_type: str
    hidden_size: int = 768

config = Config(
""").complete(line=11, column=16)
    completion_names = [c.name for c in completions]
    assert "model_type=" in completion_names
    assert "hidden_size=" in completion_names


def test_autocompletion_init_with_kwargs():
    # Create a sample script
    completions = jedi.Script("""
from dataclasses import dataclass
from huggingface_hub.dataclasses import strict

@strict(accept_kwargs=True)
@dataclass
class Config:
    model_type: str
    hidden_size: int = 768

config = Config(
""").complete(line=11, column=16)
    completion_names = [c.name for c in completions]
    assert "model_type=" in completion_names
    assert "hidden_size=" in completion_names


def test_strict_requires_dataclass():
    with pytest.raises(StrictDataclassDefinitionError):

        @strict
        class InvalidConfig:
            model_type: str


class TestClassValidation:
    @strict
    @dataclass
    class ParentConfig:
        foo: str = "bar"
        foo_length: int = 3

        def validate_foo_length(self):
            if len(self.foo) != self.foo_length:
                raise ValueError(f"foo must be {self.foo_length} characters long, got {len(self.foo)}")

    @strict
    @dataclass
    class ChildConfig(ParentConfig):
        number: int = 42

        def validate_number_multiple_of_foo_length(self):
            if self.number % self.foo_length != 0:
                raise ValueError(f"number must be a multiple of foo_length ({self.foo_length}), got {self.number}")

    @strict
    @dataclass
    class OtherChildConfig(ParentConfig):
        number: int = 42

    @strict
    @dataclass
    class ChildConfigWithPostInit(ParentConfig):
        def __post_init__(self):
            # Let's assume post_init doubles each value
            # Validation is ran AFTER __post_init__
            self.foo = self.foo * 2
            self.foo_length = self.foo_length * 2

    def test_parent_config_validation(self):
        # Test valid initialization
        config = self.ParentConfig(foo="bar", foo_length=3)
        assert config.foo == "bar"
        assert config.foo_length == 3

        # Test invalid initialization
        with pytest.raises(StrictDataclassClassValidationError):
            self.ParentConfig(foo="bar", foo_length=4)

    def test_child_config_validation(self):
        # Test valid initialization
        config = self.ChildConfig(foo="bar", foo_length=3, number=42)
        assert config.foo == "bar"
        assert config.foo_length == 3
        assert config.number == 42

        # Test invalid initialization
        with pytest.raises(StrictDataclassClassValidationError):
            self.ChildConfig(foo="bar", foo_length=4, number=40)

        with pytest.raises(StrictDataclassClassValidationError):
            self.ChildConfig(foo="bar", foo_length=3, number=43)

    def test_other_child_config_validation(self):
        # Test valid initialization
        config = self.OtherChildConfig(foo="bar", foo_length=3, number=43)
        assert config.foo == "bar"
        assert config.foo_length == 3
        assert config.number == 43  # not validated => did not fail

        # Test invalid initialization
        with pytest.raises(StrictDataclassClassValidationError):
            self.OtherChildConfig(foo="bar", foo_length=4, number=42)

    def test_validate_after_init(self):
        # Test valid initialization
        config = self.ParentConfig(foo="bar", foo_length=3)

        # Attributes can be updated after initialization
        config.foo = "abcd"
        config.foo_length = 4
        config.validate()  # Explicit call required

        # Explicit validation fails
        config.foo_length = 5
        with pytest.raises(StrictDataclassClassValidationError):
            config.validate()

    def test_validation_runs_after_post_init(self):
        config = self.ChildConfigWithPostInit(foo="bar", foo_length=3)
        assert config.foo == "barbar"
        assert config.foo_length == 6

        with pytest.raises(StrictDataclassClassValidationError, match="foo must be 4 characters long, got 6"):
            # post init doubles the value and then the validation fails
            self.ChildConfigWithPostInit(foo="bar", foo_length=2)


class TestClassValidationWithInheritance:
    """Regression test.

    If parent class is not a strict dataclass but defines validators, the child class should validate them too.
    """

    class Base:
        def validate_foo(self):
            if self.foo < 0:
                raise ValueError("foo must be positive")

    @strict
    @dataclass
    class Config(Base):
        foo: int
        bar: int

        def validate_bar(self):
            if self.bar < 0:
                raise ValueError("bar must be positive")

    def test_class_validation_with_inheritance(self):
        # Test valid initialization
        config = self.Config(foo=0, bar=0)
        assert config.foo == 0
        assert config.bar == 0

        # Test invalid initialization
        with pytest.raises(StrictDataclassClassValidationError):
            self.Config(foo=0, bar=-1)  # validation from child class

        with pytest.raises(StrictDataclassClassValidationError):
            self.Config(foo=-1, bar=0)  # validation from parent class


class TestClassValidateAlreadyExists:
    """Regression test.

    If a class already has a validate method, it should raise a StrictDataclassDefinitionError.
    """

    def test_validate_already_defined_by_class(self):
        with pytest.raises(StrictDataclassDefinitionError):

            @strict
            @dataclass
            class Config:
                foo: int = 0

                def validate(self):
                    pass  # already defined => should raise an error

    def test_validate_already_defined_by_parent(self):
        with pytest.raises(StrictDataclassDefinitionError):

            class ParentClass:
                def validate(self):
                    pass

            @strict
            @dataclass
            class ConfigWithParent(ParentClass):  # 'validate' already defined => should raise an error
                foo: int = 0


class ConfigDict(TypedDict):
    str_value: str
    positive_int_value: Annotated[int, positive_int]
    forward_ref_value: "ForwardDtype"
    optional_value: Optional[int]


class ConfigDictIncomplete(TypedDict, total=False):
    str_value: str
    positive_int_value: Annotated[int, positive_int]
    forward_ref_value: "ForwardDtype"
    optional_value: Optional[int]


@pytest.mark.parametrize(
    "data",
    [
        # All values are valid
        {"str_value": "foo", "positive_int_value": 1, "forward_ref_value": "bar", "optional_value": 0},
    ],
)
def test_typed_dict_valid_data(data: dict):
    validate_typed_dict(ConfigDict, data)
    validate_typed_dict(ConfigDictIncomplete, data)


@pytest.mark.parametrize(
    "data",
    [
        # Optional value cannot be omitted
        {"str_value": "foo", "positive_int_value": 1, "forward_ref_value": "bar"},
        # Other fields neither
        {"positive_int_value": 1, "forward_ref_value": "bar", "optional_value": 0},
        # Not a string
        {"str_value": 123, "positive_int_value": 1, "forward_ref_value": "bar", "optional_value": 0},
        # Not an integer
        {"str_value": "foo", "positive_int_value": "1", "forward_ref_value": "bar", "optional_value": 0},
        # Annotated validator is used
        {"str_value": "foo", "positive_int_value": -1, "forward_ref_value": "bar", "optional_value": 0},
    ],
)
def test_typed_dict_invalid_data(data: dict):
    with pytest.raises(StrictDataclassFieldValidationError):
        validate_typed_dict(ConfigDict, data)


def test_typed_dict_error_message():
    with pytest.raises(StrictDataclassFieldValidationError) as exception:
        validate_typed_dict(
            ConfigDict, {"str_value": 123, "positive_int_value": 1, "forward_ref_value": "bar", "optional_value": 0}
        )
    assert "Validation error for field 'str_value'" in str(exception.value)
    assert "Field 'str_value' expected str, got int (value: 123)" in str(exception.value)


def test_typed_dict_unknown_attribute():
    with pytest.raises(TypeError):
        validate_typed_dict(
            ConfigDict,
            {
                "str_value": "foo",
                "positive_int_value": 1,
                "forward_ref_value": "bar",
                "optional_value": 0,
                "another_value": 0,
            },
        )


def test_typed_dict_to_dataclass_is_cached():
    strict_cls = _build_strict_cls_from_typed_dict(ConfigDict)
    strict_cls_bis = _build_strict_cls_from_typed_dict(ConfigDict)
    assert strict_cls is strict_cls_bis  # "is" because dataclass is built only once


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Requires Python 3.11+")
class TestConfigDictNotRequired:
    def __init__(self):
        # cannot be defined at class level because of Python<3.11
        self.ConfigDictNotRequired = TypedDict(
            "ConfigDictNotRequired",
            {"required_value": Required[int], "not_required_value": NotRequired[int]},
            total=False,
        )

    @pytest.mark.parametrize(
        "data",
        [
            {"required_value": 1, "not_required_value": 2},
            {"required_value": 1},  # not required value is not validated
        ],
    )
    def test_typed_dict_not_required_valid_data(self, data: dict):
        validate_typed_dict(self.ConfigDictNotRequired, data)

    @pytest.mark.parametrize(
        "data",
        [
            # Missing required value
            {"not_required_value": 2},
            # If exists, the value is validated
            {"required_value": 1, "not_required_value": "2"},
        ],
    )
    def test_typed_dict_not_required_invalid_data(self, data: dict):
        with pytest.raises(StrictDataclassFieldValidationError):
            validate_typed_dict(self.ConfigDictNotRequired, data)


def test_typed_dict_total_true():
    ConfigDictTotalTrue = TypedDict("ConfigDictTotalTrue", {"value": int}, total=True)
    validate_typed_dict(ConfigDictTotalTrue, {"value": 1})
    with pytest.raises(StrictDataclassFieldValidationError):
        validate_typed_dict(ConfigDictTotalTrue, {})


def test_typed_dict_total_false():
    ConfigDictTotalFalse = TypedDict("ConfigDictTotalFalse", {"value": int}, total=False)
    validate_typed_dict(ConfigDictTotalFalse, {})
    validate_typed_dict(ConfigDictTotalFalse, {"value": 1})
