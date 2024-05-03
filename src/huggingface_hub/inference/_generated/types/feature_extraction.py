# Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from dataclasses import dataclass
from typing import List, Optional, Union

from .base import BaseInferenceType


@dataclass
class FeatureExtractionInput(BaseInferenceType):
    """Feature Extraction Input.
    Auto-generated from TEI specs.
    For more details, check out
    https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-tei-import.ts.
    """

    inputs: Union[List[str], str]
    normalize: Optional[bool] = None
    truncate: Optional[bool] = None
