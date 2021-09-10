from enum import Enum
from typing import Dict

import numpy as np
import torch
from app.pipelines import Pipeline
from huggingface_hub import HfApi
from speechbrain.pretrained import EncoderDecoderASR


class ModelType(Enum):
    ENCODER_ASR = 1
    ENCODER_DECODER_ASR = 2


def interface_to_type(interface_str):
    if interface_str == "EncoderASR":
        return ModelType.ENCODER_ASR
    elif interface_str == "EncoderDecoderASR":
        return ModelType.ENCODER_DECODER_ASR
    else:
        raise ValueError(
            f"Invalid interface: {interface_str} for Automatic Speech Recognition."
        )


def get_type(model_id):
    info = HfApi().model_info(repo_id=model_id)
    if info.config:
        if hasattr(info.config, "speechbrain"):
            interface_str = info.config["speechbrain"].get(
                "interface", "EncoderDecoderASR"
            )
        else:
            interface_str = "EncoderDecoderASR"
    else:
        interface_str = "EncoderDecoderASR"
    return interface_to_type(interface_str)


class AutomaticSpeechRecognitionPipeline(Pipeline):
    def __init__(self, model_id: str):
        model_type = get_type(model_id)
        if model_type == ModelType.ENCODER_ASR:
            # TODO: Change once Speechbrain has new release
            self.model = EncoderDecoderASR.from_hparams(source=model_id)
        elif model_type == ModelType.ENCODER_DECODER_ASR:
            self.model = EncoderDecoderASR.from_hparams(source=model_id)

        # Reduce latency
        self.model.modules.decoder.beam_size = 1

        # Please define a `self.sampling_rate` for this pipeline
        # to automatically read the input correctly
        self.sampling_rate = 16000

    def __call__(self, inputs: np.array) -> Dict[str, str]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default at 16KHz.
                Check `app.validation` if a different sample rate is required
                or if it depends on the model
        Return:
            A :obj:`dict`:. The object return should be liked {"text": "XXX"} containing
            the detected langage from the input audio
        """
        batch = torch.from_numpy(inputs).unsqueeze(0)
        rel_length = torch.tensor([1.0])
        predicted_words, predicted_tokens = self.model.transcribe_batch(
            batch, rel_length
        )
        return {"text": predicted_words[0]}
