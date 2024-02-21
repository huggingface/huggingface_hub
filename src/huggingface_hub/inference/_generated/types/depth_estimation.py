# Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# Using ./src/scripts/inference-codegen
#
# See https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import BaseInferenceType


@dataclass
class DepthEstimationInput(BaseInferenceType):
    """Inputs for Depth Estimation inference"""

    inputs: Any
    """The input image data"""
    parameters: Optional[Dict[str, Any]]
    """Additional inference parameters"""


@dataclass
class DepthEstimationOutput(BaseInferenceType):
    """Outputs of inference for the Depth Estimation task"""

    depth: Any
    """The predicted depth as an image"""
    predicted_depth: Any
    """The predicted depth as a tensor"""
