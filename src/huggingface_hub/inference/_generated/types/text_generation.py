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

    best_of: Optional[int] = None
    """The number of sampling queries to run. Only the best one (in terms of total logprob) will
    be returned.
    """
    decoder_input_details: Optional[bool] = None
    """Whether or not to output decoder input details"""
    details: Optional[bool] = None
    """Whether or not to output details"""
    do_sample: Optional[bool] = None
    """Whether to use logits sampling instead of greedy decoding when generating new tokens."""
    max_new_tokens: Optional[int] = None
    """The maximum number of tokens to generate."""
    repetition_penalty: Optional[float] = None
    """The parameter for repetition penalty. A value of 1.0 means no penalty. See [this
    paper](https://hf.co/papers/1909.05858) for more details.
    """
    return_full_text: Optional[bool] = None
    """Whether to prepend the prompt to the generated text."""
    seed: Optional[int] = None
    """The random sampling seed."""
    stop_sequences: Optional[List[str]] = None
    """Stop generating tokens if a member of `stop_sequences` is generated."""
    temperature: Optional[float] = None
    """The value used to modulate the logits distribution."""
    top_k: Optional[int] = None
    """The number of highest probability vocabulary tokens to keep for top-k-filtering."""
    top_p: Optional[float] = None
    """If set to < 1, only the smallest set of most probable tokens with probabilities that add
    up to `top_p` or higher are kept for generation.
    """
    truncate: Optional[int] = None
    """Truncate input tokens to the given size."""
    typical_p: Optional[float] = None
    """Typical Decoding mass. See [Typical Decoding for Natural Language
    Generation](https://hf.co/papers/2202.00666) for more information
    """
    watermark: Optional[bool] = None
    """Watermarking with [A Watermark for Large Language Models](https://hf.co/papers/2301.10226)"""


@dataclass
class TextGenerationInput(BaseInferenceType):
    """Inputs for Text Generation inference"""

    inputs: str
    """The text to initialize generation with"""
    parameters: Optional[TextGenerationParameters] = None
    """Additional inference parameters"""
    stream: Optional[bool] = None
    """Whether to stream output tokens"""


TextGenerationFinishReason = Literal["length", "eos_token", "stop_sequence"]


@dataclass
class TextGenerationPrefillToken(BaseInferenceType):
    id: int
    logprob: float
    text: str
    """The text associated with that token"""


@dataclass
class TextGenerationOutputToken(BaseInferenceType):
    """Generated token."""

    id: int
    special: bool
    """Whether or not that token is a special one"""
    text: str
    """The text associated with that token"""
    logprob: Optional[float] = None


@dataclass
class TextGenerationOutputSequenceDetails(BaseInferenceType):
    finish_reason: "TextGenerationFinishReason"
    generated_text: str
    """The generated text"""
    generated_tokens: int
    """The number of generated tokens"""
    prefill: List[TextGenerationPrefillToken]
    tokens: List[TextGenerationOutputToken]
    """The generated tokens and associated details"""
    seed: Optional[int] = None
    """The random seed used for generation"""
    top_tokens: Optional[List[List[TextGenerationOutputToken]]] = None
    """Most likely tokens"""


@dataclass
class TextGenerationOutputDetails(BaseInferenceType):
    """When enabled, details about the generation"""

    finish_reason: "TextGenerationFinishReason"
    """The reason why the generation was stopped."""
    generated_tokens: int
    """The number of generated tokens"""
    prefill: List[TextGenerationPrefillToken]
    tokens: List[TextGenerationOutputToken]
    """The generated tokens and associated details"""
    best_of_sequences: Optional[List[TextGenerationOutputSequenceDetails]] = None
    """Details about additional sequences when best_of is provided"""
    seed: Optional[int] = None
    """The random seed used for generation"""
    top_tokens: Optional[List[List[TextGenerationOutputToken]]] = None
    """Most likely tokens"""


@dataclass
class TextGenerationOutput(BaseInferenceType):
    """Outputs for Text Generation inference"""

    generated_text: str
    """The generated text"""
    details: Optional[TextGenerationOutputDetails] = None
    """When enabled, details about the generation"""


@dataclass
class TextGenerationStreamDetails(BaseInferenceType):
    """Generation details. Only available when the generation is finished."""

    finish_reason: "TextGenerationFinishReason"
    """The reason why the generation was stopped."""
    generated_tokens: int
    """The number of generated tokens"""
    seed: int
    """The random seed used for generation"""


@dataclass
class TextGenerationStreamOutput(BaseInferenceType):
    """Text Generation Stream Output"""

    token: TextGenerationOutputToken
    """Generated token."""
    details: Optional[TextGenerationStreamDetails] = None
    """Generation details. Only available when the generation is finished."""
    generated_text: Optional[str] = None
    """The complete generated text. Only available when the generation is finished."""
    index: Optional[int] = None
    """The token index within the stream. Optional to support older clients that omit it."""
