# coding=utf-8
# Copyright 2023-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Taken from https://github.com/huggingface/text-generation-inference/tree/main/clients/python
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, validator


class TextGenerationParameters(BaseModel):
    # Activate logits sampling
    do_sample: bool = False
    # Maximum number of generated tokens
    max_new_tokens: int = 20
    # The parameter for repetition penalty. 1.0 means no penalty.
    # See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    repetition_penalty: Optional[float] = None
    # Whether to prepend the prompt to the generated text
    return_full_text: bool = False
    # Stop generating tokens if a member of `stop_sequences` is generated
    stop: List[str] = []
    # Random sampling seed
    seed: Optional[int]
    # The value used to module the logits distribution.
    temperature: Optional[float]
    # The number of highest probability vocabulary tokens to keep for top-k-filtering.
    top_k: Optional[int]
    # If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
    # higher are kept for generation.
    top_p: Optional[float]
    # truncate inputs tokens to the given size
    truncate: Optional[int]
    # Typical Decoding mass
    # See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
    typical_p: Optional[float]
    # Generate best_of sequences and return the one if the highest token logprobs
    best_of: Optional[int]
    # Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
    watermark: bool = False
    # Get generation details
    details: bool = False
    # Get decoder input token logprobs and ids
    decoder_input_details: bool = False

    @validator("best_of")
    def valid_best_of(cls, field_value, values):
        if field_value is not None:
            if field_value <= 0:
                raise ValueError("`best_of` must be strictly positive")
            if field_value > 1 and values["seed"] is not None:
                raise ValueError("`seed` must not be set when `best_of` is > 1")
            sampling = (
                values["do_sample"]
                | (values["temperature"] is not None)
                | (values["top_k"] is not None)
                | (values["top_p"] is not None)
                | (values["typical_p"] is not None)
            )
            if field_value > 1 and not sampling:
                raise ValueError("you must use sampling when `best_of` is > 1")

        return field_value

    @validator("repetition_penalty")
    def valid_repetition_penalty(cls, v):
        if v is not None and v <= 0:
            raise ValueError("`repetition_penalty` must be strictly positive")
        return v

    @validator("seed")
    def valid_seed(cls, v):
        if v is not None and v < 0:
            raise ValueError("`seed` must be positive")
        return v

    @validator("temperature")
    def valid_temp(cls, v):
        if v is not None and v <= 0:
            raise ValueError("`temperature` must be strictly positive")
        return v

    @validator("top_k")
    def valid_top_k(cls, v):
        if v is not None and v <= 0:
            raise ValueError("`top_k` must be strictly positive")
        return v

    @validator("top_p")
    def valid_top_p(cls, v):
        if v is not None and (v <= 0 or v >= 1.0):
            raise ValueError("`top_p` must be > 0.0 and < 1.0")
        return v

    @validator("truncate")
    def valid_truncate(cls, v):
        if v is not None and v <= 0:
            raise ValueError("`truncate` must be strictly positive")
        return v

    @validator("typical_p")
    def valid_typical_p(cls, v):
        if v is not None and (v <= 0 or v >= 1.0):
            raise ValueError("`typical_p` must be > 0.0 and < 1.0")
        return v


class TextGenerationRequest(BaseModel):
    # Prompt
    inputs: str
    # Generation parameters
    parameters: Optional[TextGenerationParameters]
    # Whether to stream output tokens
    stream: bool = False

    @validator("inputs")
    def valid_input(cls, v):
        if not v:
            raise ValueError("`inputs` cannot be empty")
        return v

    @validator("stream")
    def valid_best_of_stream(cls, field_value, values):
        parameters = values["parameters"]
        if parameters is not None and parameters.best_of is not None and parameters.best_of > 1 and field_value:
            raise ValueError("`best_of` != 1 is not supported when `stream` == True")
        return field_value


# Decoder input tokens
class InputToken(BaseModel):
    # Token ID from the model tokenizer
    id: int
    # Token text
    text: str
    # Logprob
    # Optional since the logprob of the first token cannot be computed
    logprob: Optional[float]


# Generated tokens
class Token(BaseModel):
    # Token ID from the model tokenizer
    id: int
    # Token text
    text: str
    # Logprob
    logprob: float
    # Is the token a special token
    # Can be used to ignore tokens when concatenating
    special: bool


# Generation finish reason
class FinishReason(str, Enum):
    # number of generated tokens == `max_new_tokens`
    Length = "length"
    # the model generated its end of sequence token
    EndOfSequenceToken = "eos_token"
    # the model generated a text included in `stop_sequences`
    StopSequence = "stop_sequence"


# Additional sequences when using the `best_of` parameter
class BestOfSequence(BaseModel):
    # Generated text
    generated_text: str
    # Generation finish reason
    finish_reason: FinishReason
    # Number of generated tokens
    generated_tokens: int
    # Sampling seed if sampling was activated
    seed: Optional[int]
    # Decoder input tokens, empty if decoder_input_details is False
    prefill: List[InputToken]
    # Generated tokens
    tokens: List[Token]


# `generate` details
class Details(BaseModel):
    # Generation finish reason
    finish_reason: FinishReason
    # Number of generated tokens
    generated_tokens: int
    # Sampling seed if sampling was activated
    seed: Optional[int]
    # Decoder input tokens, empty if decoder_input_details is False
    prefill: List[InputToken]
    # Generated tokens
    tokens: List[Token]
    # Additional sequences when using the `best_of` parameter
    best_of_sequences: Optional[List[BestOfSequence]]


# `generate` return value
class TextGenerationResponse(BaseModel):
    # Generated text
    generated_text: str
    # Generation details
    details: Details


# `generate_stream` details
class StreamDetails(BaseModel):
    # Generation finish reason
    finish_reason: FinishReason
    # Number of generated tokens
    generated_tokens: int
    # Sampling seed if sampling was activated
    seed: Optional[int]


# `generate_stream` return value
class TextGenerationStreamResponse(BaseModel):
    # Generated token
    token: Token
    # Complete generated text
    # Only available when the generation is finished
    generated_text: Optional[str]
    # Generation details
    # Only available when the generation is finished
    details: Optional[StreamDetails]
