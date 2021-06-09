from typing import Tuple, List

import numpy as np
from app.pipelines import Pipeline
from asteroid import separate
from asteroid.models import BaseModel


class AudioToAudioPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model = BaseModel.from_pretrained(model_id)
        self.sampling_rate = self.model.sample_rate

    def __call__(self, inputs: np.array) -> Tuple[np.array, int, List[str]]:
        # Pass wav as [batch, n_chan, time]; here: [1, 1, time]
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
        separated = separate.numpy_separate(self.model, inputs.reshape((1, 1, -1)))
        # FIXME: how to deal with multiple sources?
        out = separated[0]
        n = out.shape[0]
        labels = [f"label_{i}" for i in range(n)]
        return separated[0], int(self.model.sample_rate), labels
