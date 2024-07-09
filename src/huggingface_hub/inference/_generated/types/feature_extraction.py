# Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
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
