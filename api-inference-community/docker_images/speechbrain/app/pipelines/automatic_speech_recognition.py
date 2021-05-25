from typing import Dict

import numpy as np
import torch
from app.pipelines import Pipeline
from speechbrain.pretrained import EncoderDecoderASR


class AutomaticSpeechRecognitionPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model = EncoderDecoderASR.from_hparams(source=model_id)
        # Reduce latency
        self.model.modules.decoder.beam_size = 1
        # Please define a `self.sampling_rate` for this pipeline
        # to automatically read the input correctly
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
        batch = torch.from_numpy(inputs).unsqueeze(0)
        rel_length = torch.tensor([1.0])
        predicted_words, predicted_tokens = self.model.transcribe_batch(
            batch, rel_length
        )
        return {"text": predicted_words[0]}
