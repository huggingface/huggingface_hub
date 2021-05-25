from typing import Tuple

import numpy as np
from app.pipelines import Pipeline
from asteroid import separate
from asteroid.models import BaseModel


class AudioSourceSeparationPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model = BaseModel.from_pretrained(model_id)
        self.sampling_rate = self.model.sample_rate

    def __call__(self, inputs: np.array) -> Tuple[np.array, int]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default at 16KHz.
                Check `app.validation` if a different sample rate is required
                or if it depends on the model
        Return:
            A :obj:`np.array` and a :obj:`int`: The raw waveform as a numpy array, and the sampling rate as an int.
        """
        # Pass wav as [batch, n_chan, time]; here: [1, 1, time]
        separated = separate.numpy_separate(self.model, inputs.reshape((1, 1, -1)))
        # FIXME: how to deal with multiple sources?
        return separated[0, 0], int(self.model.sample_rate)
