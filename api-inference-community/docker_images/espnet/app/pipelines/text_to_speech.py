from typing import Tuple

import numpy as np
from app.pipelines import Pipeline
from espnet2.bin.tts_inference import Text2Speech


class TextToSpeechPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model = Text2Speech.from_pretrained(model_id, device="cpu")

    def __call__(self, inputs: str) -> Tuple[np.array, int]:
        """
        Args:
            inputs (:obj:`str`):
                The text to generate audio from
        Return:
            A :obj:`bytes`:. The raw audio encoded as a wav format.
        """
        text = inputs
        outputs = self.model(text)
        speech = outputs[0]
        array = speech.numpy()
        return array, self.model.fs
