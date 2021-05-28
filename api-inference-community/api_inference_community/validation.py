import json
import subprocess
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConstrainedFloat, ConstrainedInt, ConstrainedList


class MinLength(ConstrainedInt):
    ge = 1
    le = 256
    strict = True


class MaxLength(ConstrainedInt):
    ge = 1
    le = 256
    strict = True


class TopK(ConstrainedInt):
    ge = 1
    strict = True


class TopP(ConstrainedFloat):
    ge = 0.0
    le = 1.0
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


class TextGenerationParamsCheck(BaseModel):
    max_new_tokens: Optional[MaxLength] = None
    top_k: Optional[TopK] = None
    top_p: Optional[TopP] = None
    repetition_penalty: Optional[RepetitionPenalty] = None
    temperature: Optional[Temperature] = None
    return_full_text: Optional[bool] = None
    num_return_sequences: Optional[NumReturnSequences] = None


class SummarizationParamsCheck(BaseModel):
    min_length: Optional[MinLength] = None
    max_length: Optional[MaxLength] = None


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


def normalize_payload(
    bpayload: bytes, task: str, sampling_rate: Optional[int] = None
) -> Tuple[Any, Dict]:
    if task in {
        "automatic-speech-recognition",
        "audio-source-separation",
    }:
        if sampling_rate is None:
            raise EnvironmentError(
                "We cannot normalize audio file if we don't know the sampling rate"
            )
        return normalize_payload_audio(bpayload, sampling_rate)
    elif task in {
        "image-classification",
    }:
        return normalize_payload_image(bpayload)
    else:
        return normalize_payload_nlp(bpayload, task)


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

    audio = np.frombuffer(out_bytes, np.float32)
    if audio.shape[0] == 0:
        raise ValueError("Malformed soundfile")
    return audio


def normalize_payload_image(bpayload: bytes) -> Tuple[Any, Dict]:
    from PIL import Image

    img = Image.open(BytesIO(bpayload))
    return img, {}


def normalize_payload_audio(bpayload: bytes, sampling_rate: int) -> Tuple[Any, Dict]:
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
