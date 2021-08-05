from typing import Tuple

import numpy as np
from app.pipelines import Pipeline


class TextToSpeechPipeline(Pipeline):
    def __init__(self, model_id: str):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        super().__init__(model_id)

    def __call__(self, inputs: str) -> Tuple[np.array, int]:
        """
        Args:
            inputs (:obj:`str`):
                The text to generate audio from
        Return:
            A :obj:`np.array` and a :obj:`int`: The raw waveform as a numpy array, and the sampling rate as an int.
        """
        return super().__call__(inputs)
