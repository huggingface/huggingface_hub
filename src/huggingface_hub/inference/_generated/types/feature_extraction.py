# Inference code generated from the JSON schema spec in ./spec
#
# Using src/scripts/inference-codegen
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import BaseInferenceType


@dataclass
class FeatureExtractionInput(BaseInferenceType):
    """Inputs for Text Embedding inference"""

    inputs: str
    """The text to get the embeddings of"""
    parameters: Optional[Dict[str, Any]] = None
    """Additional inference parameters"""
