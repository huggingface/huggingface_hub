from typing import Tuple

import numpy as np
from app.pipelines import Pipeline
from espnet2.bin.tts_inference import Text2Speech


class TextToSpeechPipeline(Pipeline):
    def __init__(self, model_id: str):
        print("LOADING MODEL")
        self.model = Text2Speech.from_pretrained(model_id, device="cpu")

        if hasattr(self.model.fs, "sampling_rate"):
            self.sampling_rate = self.model.fs
        else:
            # 16000 by default if not specified
            self.sampling_rate = 16000

    def __call__(self, inputs: str) -> Tuple[np.array, int]:
        """
        Args:
            inputs (:obj:`str`):
                The text to generate audio from
        Return:
            A :obj:`np.array` and a :obj:`int`: The raw waveform as a numpy array, and the sampling rate as an int.
        """
        outputs = self.model(inputs)
        speech = outputs["wav"]
        return speech.numpy(), self.sampling_rate
