import inspect
from dataclasses import asdict, astuple, dataclass, is_dataclass
from typing import Any, Dict, ForwardRef, List, Literal, Optional, Set, Tuple, Union, get_type_hints

import jedi
import pytest

from huggingface_hub.dataclasses import _is_validator, as_validated_field, strict, type_validator, validated_field
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


def dtype_validation(value):
    "Torch isn't installed in runners so we will check only for `string`"
    if not isinstance(value, str):
        raise ValueError(f"Value must be string ot `torch.dtype` but got {value}")

    if isinstance(value, str) and value not in ["float32", "bfloat16", "float16"]:
        raise ValueError(f"Value must be one of `[float32, bfloat16, float16] but got {value}")


@strict
@dataclass
class ConfigForwardRef:
    dtype_ref: ForwardRef("torch.dtype")
    dtype_str: str
    dtype_validated: ForwardRef("torch.dtype") = validated_field(validator=[dtype_validation])


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


def test_forward_ref_validation():
    config = ConfigForwardRef(dtype_ref="float32", dtype_str="float32", dtype_validated="float32")
    assert config.dtype_ref == "float32"  # ideally this will be a torch.dtype object in real-life
    assert config.dtype_str == "float32"
    assert config.dtype_validated == "float32"

    # The `dtype_validated` has proper validation added in field-metadata and will be validated
    with pytest.raises(StrictDataclassFieldValidationError):
        ConfigForwardRef(dtype_ref="float32", dtype_str="float32", dtype_validated="float64")

    with pytest.raises(StrictDataclassFieldValidationError):
        ConfigForwardRef(dtype_ref="float32", dtype_str="float32", dtype_validated=-1)

    with pytest.raises(StrictDataclassFieldValidationError):
        ConfigForwardRef(dtype_ref="float32", dtype_str="float32", dtype_validated="not_dtype")

    # The `dtype_str` is a normal case and will be checked for str only
    with pytest.raises(StrictDataclassFieldValidationError):
        ConfigForwardRef(dtype_ref="float32", dtype_str=-1, dtype_validated="float32")

    # The `dtype_ref` field can be of any value and will be skipped due to `ForwardRef`
    ConfigForwardRef(dtype_ref="float64", dtype_str="float32", dtype_validated="float32")
    ConfigForwardRef(dtype_ref=-100, dtype_str="float32", dtype_validated="float32")
    ConfigForwardRef(dtype_ref=["float32"], dtype_str="float32", dtype_validated="float32")


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
        # Union types
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
        ([1, 2, 3], List[int]),
        ([1, 2, "3"], List[Union[int, str]]),
        # Tuple
        ((1, 2, 3), Tuple[int, int, int]),
        ((1, 2, "3"), Tuple[int, int, str]),
        ((1, 2, 3, 4), Tuple[int, ...]),
        # Dict
        ({"a": 1, "b": 2}, Dict[str, int]),
        ({"a": 1, "b": "2"}, Dict[str, Union[int, str]]),
        # Set
        ({1, 2, 3}, Set[int]),
        ({1, 2, "3"}, Set[Union[int, str]]),
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
            Dict[
                str,
                List[
                    Tuple[
                        int,
                        DummyClass,
                        Optional[Set[Union[int, str],]],
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
        # Union types
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
        (5, List[int]),
        ([1, 2, "3"], List[int]),
        # Tuple
        (5, Tuple[int, int, int]),
        ((1, 2, "3"), Tuple[int, int, int]),
        ((1, 2, 3, 4), Tuple[int, int, int]),
        ((1, 2, "3", 4), Tuple[int, ...]),
        # Dict
        (5, Dict[str, int]),
        ({"a": 1, "b": "2"}, Dict[str, int]),
        # Set
        (5, Set[int]),
        ({1, 2, "3"}, Set[int]),
        # Custom classes
        (5, DummyClass),
        ("John", DummyClass),
    ],
)
def test_type_validator_invalid(value, type_annotation):
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
    assert asdict(config) == {
        "model_type": "bert",
        "hidden_size": 768,
        "vocab_size": 16,
    }

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
