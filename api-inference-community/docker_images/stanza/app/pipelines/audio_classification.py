from typing import Dict, List

import numpy as np
from app.pipelines import Pipeline


class AudioClassificationPipeline(Pipeline):
    def __init__(self, model_id: str):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        # IMPLEMENT_THIS : Please define a `self.sampling_rate` for this pipeline
        # to automatically read the input correctly
        self.sampling_rate = 16000
        raise NotImplementedError(
            "Please implement AudioClassificationPipeline __init__ function"
        )

    def __call__(self, inputs: np.array) -> List[Dict[str, float]]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default at 16KHz.
        Return:
            A :obj:`list`:. The object returned should be a list like [{"label": "text", "score": 0.9939950108528137}] containing :
                - "label": A string representing what the label/class is. There can be multiple labels.
                - "score": A score between 0 and 1 describing how confident the model is for this label/class.
        """
        # IMPLEMENT_THIS
        raise NotImplementedError(
            "Please implement AudioClassificationPipeline __init__ function"
        )
