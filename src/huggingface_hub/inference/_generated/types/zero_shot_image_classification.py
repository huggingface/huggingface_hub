# Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from dataclasses import dataclass
from typing import List, Optional

from .base import BaseInferenceType


@dataclass
class ZeroShotImageClassificationParameters(BaseInferenceType):
    """Additional inference parameters for Zero Shot Image Classification"""

    candidate_labels: List[str]
    """The candidate labels for this image"""
    hypothesis_template: Optional[str] = None
    """The sentence used in conjunction with `candidate_labels` to attempt the image
    classification by replacing the placeholder with the candidate labels.
    """


@dataclass
class ZeroShotImageClassificationInput(BaseInferenceType):
    """Inputs for Zero Shot Image Classification inference"""

    inputs: str
    """The input image data to classify as a base64-encoded string."""
    parameters: ZeroShotImageClassificationParameters
<<<<<<< HEAD
    """Additional inference parameters"""
=======
    """Additional inference parameters for Zero Shot Image Classification"""
>>>>>>> 33840bea5991137603af9cfc6e142af5770a2ed7


@dataclass
class ZeroShotImageClassificationOutputElement(BaseInferenceType):
    """Outputs of inference for the Zero Shot Image Classification task"""

    label: str
    """The predicted class label."""
    score: float
    """The corresponding probability."""
