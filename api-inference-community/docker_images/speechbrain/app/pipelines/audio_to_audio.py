from enum import Enum
from typing import List, Tuple

import numpy as np
import torch
from app.pipelines import Pipeline
from huggingface_hub import HfApi
from speechbrain.pretrained import SepformerSeparation, SpectralMaskEnhancement


class ModelType(Enum):
    AUDIO_SOURCE_SEPARATION = 1
    SPEECH_ENHANCEMENT = 2


def interface_to_type(interface_str):
    if interface_str == "SepformerSeparation":
        return ModelType.AUDIO_SOURCE_SEPARATION
    elif interface_str == "SpectralMaskEnhancement":
        return ModelType.SPEECH_ENHANCEMENT
    else:
        raise ValueError(f"Invalid interface: {interface_str} for Audio to Audio.")


def get_type(model_id):
    info = HfApi().model_info(repo_id=model_id)
    if info.config:
        if "speechbrain" in info.config:
            interface_str = info.config["speechbrain"].get(
                "interface", "SepformerSeparation"
            )
        else:
            interface_str = "SepformerSeparation"
    else:
        interface_str = "SepformerSeparation"
    return interface_to_type(interface_str)


class AudioToAudioPipeline(Pipeline):
    def __init__(self, model_id: str):
        model_type = get_type(model_id)
        if model_type == ModelType.AUDIO_SOURCE_SEPARATION:
            self.model = SepformerSeparation.from_hparams(source=model_id)
            self.type = "audio-source-separation"
        elif model_type == ModelType.SPEECH_ENHANCEMENT:
            self.model = SpectralMaskEnhancement.from_hparams(source=model_id)
            self.type = "speech-enhancement"
        self.sampling_rate = self.model.hparams.sample_rate

    def __call__(self, inputs: np.array) -> Tuple[np.array, int, List[str]]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default sampled at `self.sampling_rate`.
                The shape of this array is `T`, where `T` is the time axis
        Return:
            A :obj:`tuple` containing:
              - :obj:`np.array`:
                 The return shape of the array must be `C'`x`T'`
              - a :obj:`int`: the sampling rate as an int in Hz.
              - a :obj:`List[str]`: the annotation for each out channel.
                    This can be the name of the instruments for audio source separation
                    or some annotation for speech enhancement. The length must be `C'`.
        """

        if self.type == "speech-enhancement":
            return self.enhance(inputs)
        elif self.type == "audio-source-separation":
            return self.separate(inputs)

    def separate(self, inputs):
        mix = torch.from_numpy(inputs)
        est_sources = self.model.separate_batch(mix.unsqueeze(0))
        est_sources = est_sources[0]

        # C x T
        est_sources = est_sources.transpose(1, 0)
        # normalize for loudness
        est_sources = est_sources / est_sources.abs().max(dim=1, keepdim=True).values
        n = est_sources.shape[0]
        labels = [f"label_{i}" for i in range(n)]
        return est_sources.numpy(), int(self.sampling_rate), labels

    def enhance(self, inputs: np.array):
        mix = torch.from_numpy(inputs)
        enhanced = self.model.enhance_batch(mix.unsqueeze(0))
        # C x T
        labels = ["speech_enhanced"]
        return enhanced.numpy(), int(self.sampling_rate), labels
