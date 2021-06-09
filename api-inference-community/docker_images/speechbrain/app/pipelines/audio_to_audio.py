from typing import Tuple, List

import numpy as np
import torch
from app.pipelines import Pipeline
from speechbrain.pretrained import SepformerSeparation, SpectralMaskEnhancement
import requests
import json


def get_info(model_id: str):
    ENDPOINT = "https://huggingface.co/api/models/"
    response = requests.get(f"{ENDPOINT}{model_id}")
    if response.status_code != 200:
        raise Exception("Cannot infer the code properly, please set some tags")
    model_info = json.loads(response.content.decode("utf-8"))
    tags = [tag.lower().replace(" ", "-").replace("_", "-") for tag in model_info["tags"]]
    return tags


class AudioToAudioPipeline(Pipeline):
    def __init__(self, model_id: str):
        tags = get_info(model_id)
        if "audio-source-separation" in tags:
            self.model = SepformerSeparation.from_hparams(source=model_id)
            self.type = "audio-source-separation"
        elif "speech-enhancement" in tags:
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
