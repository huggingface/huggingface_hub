# Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from dataclasses import dataclass
from typing import Any, List, Literal, Optional

from .base import BaseInferenceType


TypeEnum = Literal["json", "regex"]


@dataclass
class TextGenerationInputGrammarType(BaseInferenceType):
    type: "TypeEnum"
    value: Any
    """A string that represents a [JSON Schema](https://json-schema.org/).
    JSON Schema is a declarative language that allows to annotate JSON documents
    with types and descriptions.
    """


@dataclass
class TextGenerationInputGenerateParameters(BaseInferenceType):
    best_of: Optional[int] = None
    decoder_input_details: Optional[bool] = None
    details: Optional[bool] = None
    do_sample: Optional[bool] = None
    frequency_penalty: Optional[float] = None
    grammar: Optional[TextGenerationInputGrammarType] = None
    max_new_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    return_full_text: Optional[bool] = None
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_n_tokens: Optional[int] = None
    top_p: Optional[float] = None
    truncate: Optional[int] = None
    typical_p: Optional[float] = None
    watermark: Optional[bool] = None


@dataclass
class TextGenerationInput(BaseInferenceType):
    """Text Generation Input.
    Auto-generated from TGI specs.
    For more details, check out
    https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-tgi-import.ts.
    """

    inputs: str
    parameters: Optional[TextGenerationInputGenerateParameters] = None
    stream: Optional[bool] = None


TextGenerationOutputFinishReason = Literal["length", "eos_token", "stop_sequence"]


@dataclass
class TextGenerationOutputPrefillToken(BaseInferenceType):
    id: int
    logprob: float
    text: str


@dataclass
class TextGenerationOutputToken(BaseInferenceType):
    id: int
    logprob: float
    special: bool
    text: str


@dataclass
class TextGenerationOutputBestOfSequence(BaseInferenceType):
    finish_reason: "TextGenerationOutputFinishReason"
    generated_text: str
    generated_tokens: int
    prefill: List[TextGenerationOutputPrefillToken]
    tokens: List[TextGenerationOutputToken]
    seed: Optional[int] = None
    top_tokens: Optional[List[List[TextGenerationOutputToken]]] = None


@dataclass
class TextGenerationOutputDetails(BaseInferenceType):
    finish_reason: "TextGenerationOutputFinishReason"
    generated_tokens: int
    prefill: List[TextGenerationOutputPrefillToken]
    tokens: List[TextGenerationOutputToken]
    best_of_sequences: Optional[List[TextGenerationOutputBestOfSequence]] = None
    seed: Optional[int] = None
    top_tokens: Optional[List[List[TextGenerationOutputToken]]] = None


@dataclass
class TextGenerationOutput(BaseInferenceType):
    """Text Generation Output.
    Auto-generated from TGI specs.
    For more details, check out
    https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-tgi-import.ts.
    """

    generated_text: str
    details: Optional[TextGenerationOutputDetails] = None


@dataclass
class TextGenerationStreamOutputStreamDetails(BaseInferenceType):
    finish_reason: "TextGenerationOutputFinishReason"
    generated_tokens: int
    seed: Optional[int] = None


@dataclass
class TextGenerationStreamOutputToken(BaseInferenceType):
    id: int
    logprob: float
    special: bool
    text: str


@dataclass
class TextGenerationStreamOutput(BaseInferenceType):
    """Text Generation Stream Output.
    Auto-generated from TGI specs.
    For more details, check out
    https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-tgi-import.ts.
    """

    index: int
    token: TextGenerationStreamOutputToken
    details: Optional[TextGenerationStreamOutputStreamDetails] = None
    generated_text: Optional[str] = None
    top_tokens: Optional[List[TextGenerationStreamOutputToken]] = None
