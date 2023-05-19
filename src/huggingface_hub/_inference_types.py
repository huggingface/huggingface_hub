from typing import Any, List

from .utils._typing import TypedDict


Image = Any


class ClassificationOutput(TypedDict):
    label: str
    score: float


class ConversationalOutputConversation(TypedDict):
    generated_responses: List[str]
    past_user_inputs: List[str]


class ConversationalOutput(TypedDict):
    conversation: ConversationalOutputConversation
    generated_text: str
    warnings: List[str]


class ImageSegmentationOutput(TypedDict):
    label: str
    mask: Image
    score: float
