# Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from dataclasses import dataclass
from typing import List, Literal, Optional

from .base import BaseInferenceType


@dataclass
class TextGenerationParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Text Generation
    """

    best_of: Optional[int]
    """The number of sampling queries to run. Only the best one (in terms of total logprob) will
    be returned.
    """
    decoder_input_details: Optional[bool]
    """Whether or not to output decoder input details"""
    details: Optional[bool]
    """Whether or not to output details"""
    do_sample: Optional[bool]
    """Whether to use logits sampling instead of greedy decoding when generating new tokens."""
    max_new_tokens: Optional[int]
    """The maximum number of tokens to generate."""
    repetition_penalty: Optional[float]
    """The parameter for repetition penalty. A value of 1.0 means no penalty. See [this
    paper](https://hf.co/papers/1909.05858) for more details.
    """
    return_full_text: Optional[bool]
    """Whether to prepend the prompt to the generated text."""
    seed: Optional[int]
    """The random sampling seed."""
    stop_sequences: Optional[List[str]]
    """Stop generating tokens if a member of `stop_sequences` is generated."""
    temperature: Optional[float]
    """The value used to modulate the logits distribution."""
    top_k: Optional[int]
    """The number of highest probability vocabulary tokens to keep for top-k-filtering."""
    top_p: Optional[float]
    """If set to < 1, only the smallest set of most probable tokens with probabilities that add
    up to `top_p` or higher are kept for generation.
    """
    truncate: Optional[int]
    """Truncate input tokens to the given size."""
    typical_p: Optional[float]
    """Typical Decoding mass. See [Typical Decoding for Natural Language
    Generation](https://hf.co/papers/2202.00666) for more information
    """
    watermark: Optional[bool]
    """Watermarking with [A Watermark for Large Language Models](https://hf.co/papers/2301.10226)"""


@dataclass
class TextGenerationInput(BaseInferenceType):
    """Inputs for Text Generation inference"""

    inputs: str
    """The text to initialize generation with"""
    parameters: Optional[TextGenerationParameters]
    """Additional inference parameters"""
    stream: Optional[bool]
    """Whether to stream output tokens"""


FinishReason = Literal["length", "eos_token", "stop_sequence"]


@dataclass
class PrefillToken(BaseInferenceType):
    id: int
    logprob: float
    text: str
    """The text associated with that token"""


@dataclass
class TokenElement(BaseInferenceType):
    id: int
    logprob: float
    special: bool
    """Whether or not that token is a special one"""
    text: str
    """The text associated with that token"""


@dataclass
class TextGenerationDetails(BaseInferenceType):
    finish_reason: "FinishReason"
    """The reason why the generation was stopped."""
    generated_text: int
    """The generated text"""
    generated_tokens: int
    """The number of generated tokens"""
    prefill: List[PrefillToken]
    tokens: List[TokenElement]
    """The generated tokens and associated details"""
    seed: Optional[int]
    """The random seed used for generation"""


@dataclass
class TextGenerationOutputDetails(BaseInferenceType):
    """When enabled, details about the generation"""

    finish_reason: "FinishReason"
    """The reason why the generation was stopped."""
    generated_tokens: int
    """The number of generated tokens"""
    prefill: List[PrefillToken]
    tokens: List[TokenElement]
    """The generated tokens and associated details"""
    best_of_sequences: Optional[List[TextGenerationDetails]]
    """Details about additional sequences when best_of is provided"""
    seed: Optional[int]
    """The random seed used for generation"""


@dataclass
class TextGenerationOutput(BaseInferenceType):
    """Outputs for Text Generation inference"""

    generated_text: str
    """The generated text"""
    details: Optional[TextGenerationOutputDetails]
    """When enabled, details about the generation"""


@dataclass
class TextGenerationStreamDetails(BaseInferenceType):
    """Generation details. Only available when the generation is finished."""

    finish_reason: "FinishReason"
    """The reason why the generation was stopped."""
    generated_tokens: int
    """The number of generated tokens"""
    seed: int
    """The random seed used for generation"""


@dataclass
class TextGenerationStreamOutputToken(BaseInferenceType):
    """Generated token."""

    id: int
    logprob: float
    special: bool
    """Whether or not that token is a special one"""
    text: str
    """The text associated with that token"""


@dataclass
class TextGenerationStreamOutput(BaseInferenceType):
    """Text Generation Stream Output"""

    token: TextGenerationStreamOutputToken
    """Generated token."""
    details: Optional[TextGenerationStreamDetails]
    """Generation details. Only available when the generation is finished."""
    generated_text: Optional[str]
    """The complete generated text. Only available when the generation is finished."""
    index: Optional[int]
    """The token index within the stream. Optional to support older clients that omit it."""
