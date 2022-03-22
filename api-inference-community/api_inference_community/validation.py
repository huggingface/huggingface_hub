import json
import os
import subprocess
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import (
    BaseModel,
    ConstrainedFloat,
    ConstrainedInt,
    ConstrainedList,
    validator,
)


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

    @validator("table")
    def all_rows_must_have_same_length(cls, table: Dict[str, List[str]]):
        rows = list(table.values())
        n = len(rows[0])
        if all(len(x) == n for x in rows):
            return table
        raise ValueError("All rows in the table must be the same length")


class StructuredDataClassificationInputsCheck(BaseModel):
    data: Dict[str, List[str]]

    @validator("data")
    def all_rows_must_have_same_length(cls, data: Dict[str, List[str]]):
        rows = list(data.values())
        n = len(rows[0])
        if all(len(x) == n for x in rows):
            return data
        raise ValueError("All rows in the data must be the same length")


class StringOrStringBatchInputCheck(BaseModel):
    __root__: Union[List[str], str]

    @validator("__root__")
    def input_must_not_be_empty(cls, __root__: Union[List[str], str]):
        if isinstance(__root__, list):
            if len(__root__) == 0:
                raise ValueError(
                    "The inputs are invalid, at least one input is required"
                )
        return __root__


class StringInput(BaseModel):
    __root__: str


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
    "feature-extraction": StringOrStringBatchInputCheck,
    "sentence-similarity": SentenceSimilarityInputsCheck,
    "table-question-answering": TableQuestionAnsweringInputsCheck,
    "structured-data-classification": StructuredDataClassificationInputsCheck,
    "fill-mask": StringInput,
    "summarization": StringInput,
    "text2text-generation": StringInput,
    "text-generation": StringInput,
    "text-classification": StringInput,
    "token-classification": StringInput,
    "translation": StringInput,
    "zero-shot-classification": StringInput,
    "text-to-speech": StringInput,
    "text-to-image": StringInput,
}

BATCH_ENABLED_PIPELINES = ["feature-extraction"]


def check_params(params, tag):
    if tag in PARAMS_MAPPING:
        PARAMS_MAPPING[tag].parse_obj(params)
    return True


def check_inputs(inputs, tag):
    if tag in INPUTS_MAPPING:
        INPUTS_MAPPING[tag].parse_obj(inputs)
        return True
    else:
        raise ValueError(f"{tag} is not a valid pipeline.")


AUDIO_INPUTS = {
    "automatic-speech-recognition",
    "audio-to-audio",
    "speech-segmentation",
    "audio-classification",
}

IMAGE_INPUTS = {
    "image-classification",
    "image-segmentation",
    "image-to-text",
    "object-detection",
}

TEXT_INPUTS = {
    "conversational",
    "feature-extraction",
    "question-answering",
    "sentence-similarity",
    "fill-mask",
    "table-question-answering",
    "structured-data-classification",
    "summarization",
    "text2text-generation",
    "text-classification",
    "text-to-image",
    "text-to-speech",
    "token-classification",
    "zero-shot-classification",
}


def normalize_payload(
    bpayload: bytes, task: str, sampling_rate: Optional[int] = None
) -> Tuple[Any, Dict]:
    if task in AUDIO_INPUTS:
        if sampling_rate is None:
            raise EnvironmentError(
                "We cannot normalize audio file if we don't know the sampling rate"
            )
        outputs = normalize_payload_audio(bpayload, sampling_rate)
        return outputs
    elif task in IMAGE_INPUTS:
        return normalize_payload_image(bpayload)
    elif task in TEXT_INPUTS:
        return normalize_payload_nlp(bpayload, task)
    else:
        raise EnvironmentError(
            f"The task `{task}` is not recognized by api-inference-community"
        )


def ffmpeg_convert(array: np.array, sampling_rate: int) -> bytes:
    """
    Helper function to convert raw waveforms to actual compressed file (lossless compression here)
    """
    ar = str(sampling_rate)
    ac = "1"
    format_for_conversion = "flac"
    ffmpeg_command = [
        "ffmpeg",
        "-ac",
        "1",
        "-f",
        "f32le",
        "-ac",
        ac,
        "-ar",
        ar,
        "-i",
        "pipe:0",
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    ffmpeg_process = subprocess.Popen(
        ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    output_stream = ffmpeg_process.communicate(array.tobytes())
    out_bytes = output_stream[0]
    if len(out_bytes) == 0:
        raise Exception("Impossible to convert output stream")
    return out_bytes


def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array:
    """
    Librosa does that under the hood but forces the use of an actual
    file leading to hitting disk, which is almost always very bad.
    """
    ar = f"{sampling_rate}"
    ac = "1"
    format_for_conversion = "f32le"
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    ffmpeg_process = subprocess.Popen(
        ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    output_stream = ffmpeg_process.communicate(bpayload)
    out_bytes = output_stream[0]

    audio = np.frombuffer(out_bytes, np.float32).copy()
    if audio.shape[0] == 0:
        raise ValueError("Malformed soundfile")
    return audio


def normalize_payload_image(bpayload: bytes) -> Tuple[Any, Dict]:
    from PIL import Image

    img = Image.open(BytesIO(bpayload))
    return img, {}


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".mp4", ".webm", ".aac"}


DATA_PREFIX = os.getenv("HF_TRANSFORMERS_CACHE", "")


def normalize_payload_audio(bpayload: bytes, sampling_rate: int) -> Tuple[Any, Dict]:
    if os.path.isfile(bpayload) and bpayload.startswith(DATA_PREFIX.encode("utf-8")):
        # XXX:
        # This is necessary for batch jobs where the datasets can contain
        # filenames instead of the raw data.
        # We attempt to sanitize this roughly, by checking it lives on the data
        # path (harcoded in the deployment and in all the dockerfiles)
        # We also attempt to prevent opening files that are not obviously
        # audio files, to prevent opening stuff like model weights.
        filename, ext = os.path.splitext(bpayload)
        if ext.decode("utf-8") in AUDIO_EXTENSIONS:
            with open(bpayload, "rb") as f:
                bpayload = f.read()
    inputs = ffmpeg_read(bpayload, sampling_rate)
    if len(inputs.shape) > 1:
        # ogg can take dual channel input -> take only first input channel in this case
        inputs = inputs[:, 0]
    return inputs, {}


def normalize_payload_nlp(bpayload: bytes, task: str) -> Tuple[Any, Dict]:
    payload = bpayload.decode("utf-8")

    # We used to accept raw strings, we need to maintain backward compatibility
    try:
        payload = json.loads(payload)
    except Exception:
        pass

    parameters: Dict[str, Any] = {}
    if isinstance(payload, dict) and "inputs" in payload:
        inputs = payload["inputs"]
        parameters = payload.get("parameters", {})
    else:
        inputs = payload
    check_params(parameters, task)
    check_inputs(inputs, task)
    return inputs, parameters
