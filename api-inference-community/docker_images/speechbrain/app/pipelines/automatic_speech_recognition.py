from typing import Dict

import numpy as np
import torch
from app.common import ModelType, get_type
from app.pipelines import Pipeline
from speechbrain.pretrained import EncoderASR, EncoderDecoderASR


class AutomaticSpeechRecognitionPipeline(Pipeline):
    def __init__(self, model_id: str):
        model_type = get_type(model_id)
        if model_type is ModelType.ENCODERASR:
            self.model = EncoderASR.from_hparams(source=model_id)
        elif model_type is ModelType.ENCODERDECODERASR:
            self.model = EncoderDecoderASR.from_hparams(source=model_id)

            # Reduce latency
            self.model.mods.decoder.beam_size = 1
        else:
            raise ValueError(
                f"{model_type.value} is invalid for automatic-speech-recognition"
            )

        # Please define a `self.sampling_rate` for this pipeline
        # to automatically read the input correctly
        self.sampling_rate = self.model.hparams.sample_rate

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
