# Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from dataclasses import dataclass
from typing import Any, Literal, Optional

from .base import BaseInferenceType


ClassificationOutputTransform = Literal["sigmoid", "softmax", "none"]


@dataclass
class VideoClassificationParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Video Classification
    """

    frame_sampling_rate: Optional[int]
    """The sampling rate used to select frames from the video."""
    function_to_apply: Optional["ClassificationOutputTransform"]
    num_frames: Optional[int]
    """The number of sampled frames to consider for classification."""
    top_k: Optional[int]
    """When specified, limits the output to the top K most probable classes."""


@dataclass
class VideoClassificationInput(BaseInferenceType):
    """Inputs for Video Classification inference"""

    inputs: Any
    """The input video data"""
    parameters: Optional[VideoClassificationParameters]
    """Additional inference parameters"""


@dataclass
class VideoClassificationOutput(BaseInferenceType):
    """Outputs of inference for the Video Classification task"""

    label: str
    """The predicted class label."""
    score: float
    """The corresponding probability."""
