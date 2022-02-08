import datetime
from typing import Dict

import numpy as np
import torch
from app.pipelines import Pipeline
from pyannote.audio import Pipeline as Pypeline


class AutomaticSpeechRecognitionPipeline(Pipeline):
    def __init__(self, model_id: str):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        # IMPLEMENT_THIS : Please define a `self.sampling_rate` for this pipeline
        # to automatically read the input correctly
        self.sampling_rate = 16000
        self.model = Pypeline.from_pretrained(model_id)

    def __call__(self, inputs: np.array) -> Dict[str, str]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default at self.sampling_rate, otherwise 16KHz.
        Return:
            A :obj:`dict`:. The object return should be liked {"text": "XXX"} containing
            the detected langage from the input audio
        """
        wav = torch.from_numpy(inputs).unsqueeze(0)
        output = self.model({"waveform": wav, "sample_rate": self.sampling_rate})
        regions = "".join(
            [
                f"|{str(datetime.timedelta(seconds=segment.start))[:-3]} - {str(datetime.timedelta(seconds=segment.end))[:-3]} : {label} |"
                for segment, _, label in output.itertracks(yield_label=True)
            ]
        )
        return {"text": regions}
