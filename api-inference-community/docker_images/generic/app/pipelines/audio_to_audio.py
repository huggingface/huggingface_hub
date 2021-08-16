from typing import List, Tuple

import numpy as np
from app.pipelines import Pipeline


class AudioToAudioPipeline(Pipeline):
    def __init__(self, model_id: str):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        # IMPLEMENT_THIS : Please define a `self.sampling_rate` for this pipeline
        # to automatically read the input correctly
        self.sampling_rate = 16000
        raise NotImplementedError(
            "Please implement AudioToAudioPipeline __init__ function"
        )

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
        # IMPLEMENT_THIS
        raise NotImplementedError(
            "Please implement AudioToAudioPipeline __call__ function"
        )
