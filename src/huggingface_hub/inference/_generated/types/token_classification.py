# Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from dataclasses import dataclass
from typing import Any, List, Literal, Optional

from .base import BaseInferenceType


TokenClassificationAggregationStrategy = Literal["none", "simple", "first", "average", "max"]


@dataclass
class TokenClassificationParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Token Classification
    """

    aggregation_strategy: Optional["TokenClassificationAggregationStrategy"]
    """The strategy used to fuse tokens based on model predictions"""
    ignore_labels: Optional[List[str]]
    """A list of labels to ignore"""
    stride: Optional[int]
    """The number of overlapping tokens between chunks when splitting the input text."""


@dataclass
class TokenClassificationInput(BaseInferenceType):
    """Inputs for Token Classification inference"""

    inputs: str
    """The input text data"""
    parameters: Optional[TokenClassificationParameters]
    """Additional inference parameters"""


@dataclass
class TokenClassificationOutputElement(BaseInferenceType):
    """Outputs of inference for the Token Classification task"""

    label: Any
    score: float
    """The associated score / probability"""
    end: Optional[int]
    """The character position in the input where this group ends."""
    entity_group: Optional[str]
    """The predicted label for that group of tokens"""
    start: Optional[int]
    """The character position in the input where this group begins."""
    word: Optional[str]
    """The corresponding text"""
