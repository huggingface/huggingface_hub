from typing import Dict

import numpy as np
from app.pipelines import Pipeline


class SpeechSegmentationPipeline(Pipeline):
    def __init__(self, model_id: str):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        # IMPLEMENT_THIS : Please define a `self.sampling_rate` for this pipeline
        # to automatically read the input correctly
        self.sampling_rate = 16000
        raise NotImplementedError(
            "Please implement SpeechSegmentationPipeline __init__ function"
        )

    def __call__(self, inputs: np.array) -> Dict[str, str]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default at self.sampling_rate, otherwise 16KHz.
        Return:
            A :obj:`list`:. Each item in the list is like {"class": "XXX", "start": float, "end": float}
            "class" is the associated class of the audio segment, "start" and "end" are markers expressed in seconds
            within the audio file.
        """
        # IMPLEMENT_THIS
        # api_inference_community.normalizers.speaker_diarization_normalize could help.
        raise NotImplementedError(
            "Please implement SpeechSegmentationPipeline __call__ function"
        )
