import os
import subprocess
import sys
from typing import Dict, List, Union

import numpy as np
from api_inference_community.normalizers import speaker_diarization_normalize
from app.pipelines import Pipeline
from huggingface_hub import snapshot_download


class SpeechClassificationPipeline(Pipeline):
    def __init__(self, model_id: str):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        # IMPLEMENT_THIS : Please define a `self.sampling_rate` for this pipeline
        # to automatically read the input correctly
        filepath = snapshot_download(model_id)
        sys.path.append(filepath)
        if "requirements.txt" in os.listdir(filepath):
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    os.path.join(filepath, "requirements.txt"),
                ]
            )

        from model import PreTrainedModel

        self.model = PreTrainedModel(filepath)
        self.sampling_rate = 16000

    def __call__(self, inputs: np.array) -> List[Dict[str, Union[str, float]]]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default at self.sampling_rate, otherwise 16KHz.
        Return:
            A :obj:`list`:. Each item in the list is like {"class": "XXX", "start": float, "end": float}
            "class" is the associated class of the audio segment, "start" and "end" are markers expressed in seconds
            within the audio file.
        """
        # S x N boolean tensor
        # S : sequence_length
        # N : Number of expected speakers
        # Filled with ones where speaker-n is speaking
        outputs = self.model(inputs)
        return speaker_diarization_normalize(
            outputs, self.sampling_rate, ["SPEAKER_0", "SPEAKER_1"]
        )
