# Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from dataclasses import dataclass
from typing import List, Literal, Optional

from .base import BaseInferenceType


Role = Literal["system", "user", "assistant"]


@dataclass
class ChatCompletionInputMessageElement(BaseInferenceType):
    role: "Role"
    """The role of the messages author, in this case system.
    The role of the messages author, in this case user.
    The role of the messages author, in this case assistant.
    """
    content: Optional[str]
    """The contents of the system message.
    The contents of the user message.
    The contents of the assistant message.
    """


@dataclass
class ChatCompletionInput(BaseInferenceType):
    """Inputs for ChatCompletion inference"""

    frequency_penalty: Optional[float]
    """Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing
    frequency in the text so far, decreasing the model's likelihood to repeat the same line
    verbatim.
    """
    max_tokens: Optional[int]
    """The maximum number of tokens that can be generated in the chat completion. The total
    length of input tokens and generated tokens is limited by the model's context length.
    """
    messages: Optional[List[ChatCompletionInputMessageElement]]
    model: Optional[str]
    """ID of the model to use. See the model endpoint compatibility table for details on which
    models work with the Chat API.
    """
    seed: Optional[int]
    """The random sampling seed."""
    stop: Optional[str]
    """Up to 4 sequences where the API will stop generating further tokens."""
    stream: Optional[bool]
    """If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as
    data-only server-sent events as they become available, with the stream terminated by a
    data: [DONE] message.
    """
    temperature: Optional[float]
    """What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the
    output more random, while lower values like 0.2 will make it more focused and
    deterministic. We generally recommend altering this or top_p but not both.
    """
    top_p: Optional[float]
    """An alternative to sampling with temperature, called nucleus sampling, where the model
    considers the results of the tokens with top_p probability mass. So 0.1 means only the
    tokens comprising the top 10% probability mass are considered. We generally recommend
    altering this or temperature but not both.
    """


@dataclass
class ChatCompletionOutputChoiceMessage(BaseInferenceType):
    content: str
    """The content of the chat completion message."""


@dataclass
class ChatCompletionOutputChoice(BaseInferenceType):
    finish_reason: str
    """The reason the model stopped generating tokens."""
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
    model: str
    """The model used for the chat completion."""
