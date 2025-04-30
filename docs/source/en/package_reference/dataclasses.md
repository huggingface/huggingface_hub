# Strict Dataclasses

The `huggingface_hub` package provides a utility to create **strict dataclasses**. These are enhanced versions of Python's standard `dataclass` with additional validation features. Strict dataclasses ensure that fields are validated both during initialization and assignment, making them useful for scenarios where data integrity is critical.

## Overview

Strict dataclasses are created using the `@strict` decorator. They extend the functionality of regular dataclasses by:

- Validating field types based on type hints.
- Supporting custom validators for additional checks.
- Optionally allowing arbitrary keyword arguments in the constructor.
- Validating fields on initialization and on field assignment. 

## Benefits

- **Data Integrity**: Ensures that fields always contain valid data.
- **Ease of Use**: Integrates seamlessly with Python's `dataclass` module.
- **Customizability**: Supports custom validators for complex validation logic.
- **Lightweight**: do not require any additional dependency like Pydantic, attrs, etc.

## Usage

### Basic Example

```python
from dataclasses import dataclass
from huggingface_hub.dataclasses import strict, validated_field

# Custom validator to ensure a value is positive
def positive_int(value: int):
    if not value >= 0:
        raise ValueError(f"Value must be positive, got {value}")

@strict
@dataclass
class Config:
    model_type: str
    hidden_size: int = validated_field(validator=positive_int)
    vocab_size: int = 16  # Default value
```

Fields are validated during initialization:

```python
config = Config(model_type="bert", hidden_size=768)  # Valid
config = Config(model_type="bert", hidden_size=-1)   # Raises StrictDataclassFieldValidationError
```

Fields are also validated during assignment:

```python
config.hidden_size = 512  # Valid
config.hidden_size = -1   # Raises StrictDataclassFieldValidationError
```

### Custom Validators

You can attach multiple custom validators to fields using `validated_field`. A validator is a callable that takes a single argument and raises an exception if the value is invalid.

```python
def multiple_of_64(value: int):
    if value % 64 != 0:
        raise ValueError(f"Value must be a multiple of 64, got {value}")

@strict
@dataclass
class Config:
    hidden_size: int = validated_field(validator=[positive_int, multiple_of_64])
```

In this example, both validators are applied to the `hidden_size` field.

### Additional Keyword Arguments

By default, strict dataclasses only accept fields defined in the class. You can allow additional keyword arguments by setting `accept_kwargs=True` in the `@strict` decorator.

```python
@strict(accept_kwargs=True)
@dataclass
class ConfigWithKwargs:
    model_type: str
    vocab_size: int = 16

config = ConfigWithKwargs(model_type="bert", vocab_size=30000, extra_field="extra_value")
print(config)  # ConfigWithKwargs(model_type='bert', vocab_size=30000, *extra_field='extra_value')
```

Additional keyword arguments are shown in the representation of the dataclass but prefixed with `*` to highlight that they are not validated.

### Integration with Type Hints

Strict dataclasses respect type hints and validate them automatically. For example:

```python
from typing import List

@strict
@dataclass
class Config:
    layers: List[int]

config = Config(layers=[64, 128])  # Valid
config = Config(layers="not_a_list")  # Raises StrictDataclassFieldValidationError
```

Supported types:
- Any
- Union
- Optional
- Literal
- List
- Dict
- Tuple
- Set

And any combination of them.

## API Reference

### `@strict`

The `@strict` decorator enhances a dataclass with strict validation.

[[autodoc]] dataclasses.strict

### `validated_field`

Creates a dataclass field with custom validation.

[[autodoc]] dataclasses.validated_field

### Errors

[[autodoc]] errors.StrictDataclassError

[[autodoc]] errors.StrictDataclassDefinitionError

[[autodoc]] errors.StrictDataclassFieldValidationError

## Why not chose `pydantic` ? (or `attrs`? or `marshmallow_dataclass`?)

- See discussion in https://github.com/huggingface/transformers/issues/36329 related to adding pydantic as a new dependency. Would be an heavy addition + require careful logic to support both v1 and v2.
- we do not want most of pydantic's features, especially the ones related to automatic casting, jsonschema, serializations, aliases, ...
- we do not need to be able to instantiate a class from a dictionary
- we do not want to mutate data. In `@strict`, "validation" refers to "checking if a value is valid". In Pydantic, "validation" refers to "casting a value, possibly mutating it and then check if it's valid".
- we do not need blazing fast validation. `@strict` is not meant for heavy load where performances is critical. Common use case will be to validate a model configuration (only done once and very neglectable compared to running a model). This allows us to keep code minimal.