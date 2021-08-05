from typing import Dict

import numpy as np
from app.pipelines import Pipeline


class AutomaticSpeechRecognitionPipeline(Pipeline):
    def __init__(self, model_id: str):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        super().__init__(model_id)

    def __call__(self, inputs: np.array) -> Dict[str, str]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default at self.sampling_rate, otherwise 16KHz.
        Return:
            A :obj:`dict`:. The object return should be liked {"text": "XXX"} containing
            the detected langage from the input audio
        """
        return super().__call__(inputs)
