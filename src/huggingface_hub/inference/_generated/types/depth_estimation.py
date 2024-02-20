# Inference code generated from the JSON schema spec in ./spec
#
# Using src/scripts/inference-codegen
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import BaseInferenceType


@dataclass
class DepthEstimationInput(BaseInferenceType):
    """Inputs for Depth Estimation inference"""

    inputs: Any
    """The input image data"""
    parameters: Optional[Dict[str, Any]] = None
    """Additional inference parameters"""


@dataclass
class DepthEstimationOutput(BaseInferenceType):
    """Outputs of inference for the Depth Estimation task"""

    depth: Any
    """The predicted depth as an image"""
    predicted_depth: Any
    """The predicted depth as a tensor"""
