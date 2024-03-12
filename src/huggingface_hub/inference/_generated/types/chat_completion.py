# Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from dataclasses import dataclass
from typing import List, Literal, Optional, Union

from .base import BaseInferenceType


Role = Literal["assistant", "system", "user"]


@dataclass
class ChatCompletionInputMessage(BaseInferenceType):
    content: str
    """The content of the message."""
    role: "Role"
    """The role of the messages author."""


@dataclass
class ChatCompletionInput(BaseInferenceType):
    """Inputs for ChatCompletion inference"""

    messages: List[ChatCompletionInputMessage]
    frequency_penalty: Optional[float]
    """Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing
    frequency in the text so far, decreasing the model's likelihood to repeat the same line
    verbatim.
    """
    max_tokens: Optional[int]
    """The maximum number of tokens that can be generated in the chat completion."""
    seed: Optional[int]
    """The random sampling seed."""
    stop: Optional[Union[List[str], str]]
    """Stop generating tokens if a stop token is generated."""
    stream: Optional[bool]
    """If set, partial message deltas will be sent."""
    temperature: Optional[float]
    """The value used to modulate the logits distribution."""
    top_p: Optional[float]
    """If set to < 1, only the smallest set of most probable tokens with probabilities that add
    up to `top_p` or higher are kept for generation.
    """


@dataclass
class ChatCompletionOutputChoiceMessage(BaseInferenceType):
    content: str
    """The content of the chat completion message."""


@dataclass
class ChatCompletionOutputChoice(BaseInferenceType):
    finish_reason: str
    """The reason why the model stopped generating tokens."""
    index: int
    """The index of the choice in the list of choices."""
    message: ChatCompletionOutputChoiceMessage


@dataclass
class ChatCompletionOutput(BaseInferenceType):
    """Outputs for Chat Completion inference"""

    choices: List[ChatCompletionOutputChoice]
    """A list of chat completion choices."""
    created: int
    """The Unix timestamp (in seconds) of when the chat completion was created."""
