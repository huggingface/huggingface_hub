# Inference code generated from the JSON schema spec in ./spec
#
# Using src/scripts/inference-codegen
from dataclasses import dataclass
from typing import Any, Optional

from .base import BaseInferenceType


@dataclass
class VisualQuestionAnsweringInputData(BaseInferenceType):
    """One (image, question) pair to answer"""

    image: Any
    """The image."""
    question: Any
    """The question to answer based on the image."""


@dataclass
class VisualQuestionAnsweringParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Visual Question Answering
    """

    top_k: Optional[int] = None
    """The number of answers to return (will be chosen by order of likelihood). Note that we
    return less than topk answers if there are not enough options available within the
    context.
    """


@dataclass
class VisualQuestionAnsweringInput(BaseInferenceType):
    """Inputs for Visual Question Answering inference"""

    inputs: VisualQuestionAnsweringInputData
    """One (image, question) pair to answer"""
    parameters: Optional[VisualQuestionAnsweringParameters] = None
    """Additional inference parameters"""


@dataclass
class VisualQuestionAnsweringOutputElement(BaseInferenceType):
    """Outputs of inference for the Visual Question Answering task"""

    label: Any
    score: float
    """The associated score / probability"""
    answer: Optional[str] = None
    """The answer to the question"""
