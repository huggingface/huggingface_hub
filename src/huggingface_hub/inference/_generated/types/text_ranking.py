# Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.

from .base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class TextRankingInputData(BaseInferenceType):
    query: str
    """The query to rank the documents against."""
    texts: list[str]
    """The documents to rank."""


@dataclass_with_extra
class TextRankingInput(BaseInferenceType):
    """Inputs for Text Ranking inference"""

    inputs: TextRankingInputData


@dataclass_with_extra
class TextRankingOutputElement(BaseInferenceType):
    """Documents ranked by relevance to the query, in descending score order."""

    index: int
    """The index of the document in the input texts."""
    score: float
    """The relevance score of the document."""
