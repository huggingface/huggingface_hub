from typing import Dict

import numpy as np
from app.pipelines import Pipeline
from espnet2.bin.asr_inference import Speech2Text


class AutomaticSpeechRecognitionPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model = Speech2Text.from_pretrained(model_id, device="cpu")
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
        outputs = self.model(inputs)
        text, *_ = outputs[0]
        return {"text": text}
