# Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# Using ./src/scripts/inference-codegen
#
# See https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from dataclasses import dataclass
from typing import Literal, Optional

from .base import BaseInferenceType


ClassificationOutputTransform = Literal["sigmoid", "softmax", "none"]


@dataclass
class TextClassificationParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Text Classification
    """

    function_to_apply: Optional["ClassificationOutputTransform"]
    top_k: Optional[int]
    """When specified, limits the output to the top K most probable classes."""


@dataclass
class TextClassificationInput(BaseInferenceType):
    """Inputs for Text Classification inference"""

    inputs: str
    """The text to classify"""
    parameters: Optional[TextClassificationParameters]
    """Additional inference parameters"""


@dataclass
class TextClassificationOutput(BaseInferenceType):
    """Outputs of inference for the Text Classification task"""

    label: str
    """The predicted class label."""
    score: float
    """The corresponding probability."""
