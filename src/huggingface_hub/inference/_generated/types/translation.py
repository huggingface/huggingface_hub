# Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from .base import BaseInferenceType


Text2TextGenerationTruncationStrategy = Literal["do_not_truncate", "longest_first", "only_first", "only_second"]


@dataclass
class Text2TextGenerationParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Text2text Generation
    """

    clean_up_tokenization_spaces: Optional[bool]
    """Whether to clean up the potential extra spaces in the text output."""
    generate_parameters: Optional[Dict[str, Any]]
    """Additional parametrization of the text generation algorithm"""
    truncation: Optional["Text2TextGenerationTruncationStrategy"]
    """The truncation strategy to use"""


@dataclass
class TranslationInput(BaseInferenceType):
    """Inputs for Translation inference
    Inputs for Text2text Generation inference
    """

    inputs: str
    """The input text data"""
    parameters: Optional[Text2TextGenerationParameters]
    """Additional inference parameters"""


@dataclass
class TranslationOutput(BaseInferenceType):
    """Outputs for Translation inference
    Outputs of inference for the Text2text Generation task
    """

    translation_text: Any
    translation_output_translation_text: Optional[str]
    """The translated text."""
