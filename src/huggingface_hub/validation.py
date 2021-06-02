from typing import Dict, List, Optional, Union

from pydantic import (
    BaseModel,
    ConstrainedFloat,
    ConstrainedInt,
    ConstrainedList,
    validator,
)


# Subset from api-inference-community/api_inference_community/validation.py with
# some additional changes.
# TODO: Move things out of api-inference-community to a place that can be
# re-used in other places. Ideally, we could even make the typing language
# agnostic so it can be re-used by clients in different languages.


class MinLength(ConstrainedInt):
    ge = 1
    le = 500
    strict = True


class MaxLength(ConstrainedInt):
    ge = 1
    le = 500
    strict = True


class TopK(ConstrainedInt):
    ge = 1
    strict = True


class TopP(ConstrainedFloat):
    ge = 0.0
    le = 1.0
    strict = True


class MaxTime(ConstrainedFloat):
    ge = 0.0
    le = 120.0
    strict = True


class NumReturnSequences(ConstrainedInt):
    ge = 1
    le = 10
    strict = True


class RepetitionPenalty(ConstrainedFloat):
    ge = 0.0
    le = 100.0
    strict = True


class Temperature(ConstrainedFloat):
    ge = 0.0
    le = 100.0
    strict = True


class CandidateLabels(ConstrainedList):
    min_items = 1
    __args__ = [str]


class FillMaskParamsCheck(BaseModel):
    top_k: Optional[TopK] = None


class ZeroShotParamsCheck(BaseModel):
    candidate_labels: Union[str, CandidateLabels]
    multi_label: Optional[bool] = None


class SharedGenerationParams(BaseModel):
    min_length: Optional[MinLength] = None
    max_length: Optional[MaxLength] = None
    top_k: Optional[TopK] = None
    top_p: Optional[TopP] = None
    max_time: Optional[MaxTime] = None
    repetition_penalty: Optional[RepetitionPenalty] = None
    temperature: Optional[Temperature] = None

    @validator("max_length")
    def max_length_must_be_larger_than_min_length(
        cls, max_length: Optional[MinLength], values: Dict[str, Optional[str]]
    ):
        if "min_length" in values:
            if values["min_length"] is not None:
                if max_length < values["min_length"]:
                    raise ValueError("min_length cannot be larger than max_length")
        return max_length


class TextGenerationParamsCheck(SharedGenerationParams):
    return_full_text: Optional[bool] = None
    num_return_sequences: Optional[NumReturnSequences] = None


class SummarizationParamsCheck(SharedGenerationParams):
    num_return_sequences: Optional[NumReturnSequences] = None


class ConversationalInputsCheck(BaseModel):
    text: str
    past_user_inputs: List[str]
    generated_responses: List[str]


class QuestionInputsCheck(BaseModel):
    question: str
    context: str


class SentenceSimilarityInputsCheck(BaseModel):
    source_sentence: str
    sentences: List[str]


class TableQuestionAnsweringInputsCheck(BaseModel):
    table: Dict[str, List[str]]
    query: str


PARAMS_MAPPING = {
    "conversational": SharedGenerationParams,
    "fill-mask": FillMaskParamsCheck,
    "text2text-generation": TextGenerationParamsCheck,
    "text-generation": TextGenerationParamsCheck,
    "summarization": SummarizationParamsCheck,
    "zero-shot-classification": ZeroShotParamsCheck,
}

INPUTS_MAPPING = {
    "conversational": ConversationalInputsCheck,
    "question-answering": QuestionInputsCheck,
    "sentence-similarity": SentenceSimilarityInputsCheck,
    "table-question-answering": TableQuestionAnsweringInputsCheck,
}

ALL_TASKS = [
    # NLP
    "text-classification",
    "token-classification",
    "table-question-answering",
    "question-answering",
    "zero-shot-classification",
    "translation",
    "summarization",
    "conversational",
    "feature-extraction",
    "text-generation",
    "text2text-generation",
    "fill-mask",
    "sentence-similarity",
    # Audio
    "text-to-speech",
    "automatic-speech-recognition",
    "audio-source-separation",
    "voice-activity-detection",
    # Computer vision
    "image-classification",
    "object-detection",
    "image-segmentation",
]


def check_params(params, tag):
    if tag in PARAMS_MAPPING:
        PARAMS_MAPPING[tag].parse_obj(params)
    return True


def check_inputs(inputs, tag):
    if tag in INPUTS_MAPPING:
        INPUTS_MAPPING[tag].parse_obj(inputs)
    else:
        # Some tasks just expect {inputs: "str"}. Such as:
        # feature-extraction
        # fill-mask
        # text2text-generation
        # text-classification
        # text-generation
        # token-classification
        # translation
        if not isinstance(inputs, str):
            raise ValueError("The inputs is invalid, we expect a string")
    return True
