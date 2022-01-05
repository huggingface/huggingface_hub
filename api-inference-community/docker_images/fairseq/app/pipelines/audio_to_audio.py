import os
from typing import List, Tuple

import numpy as np

from app.pipelines import Pipeline
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.speech_to_text.hub_interface import S2THubInterface


class SpeechToSpeechPipeline(Pipeline):
    def __init__(self, model_id: str):
        models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
            model_id,
            arg_overrides={"config_yaml": "config.yaml"},
            cache_dir=os.getenv("HUGGINGFACE_HUB_CACHE"),
        )
        self.model = models[0].cpu()
        self.model.eval()
        cfg["task"].cpu = True
        self.task = task
        self.generator = task.build_generator([self.model], cfg)

    def __call__(self, inputs: np.array) -> Tuple[np.array, int, List[str]]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default sampled at `self.sampling_rate`.
                The shape of this array is `T`, where `T` is the time axis
        Return:
            A :obj:`tuple` containing:
              - :obj:`np.array`:
                 The return shape of the array must be `C'`x`T'`
              - a :obj:`int`: the sampling rate as an int in Hz.
              - a :obj:`List[str]`: the annotation for each out channel.
                    This can be the name of the instruments for audio source separation
                    or some annotation for speech enhancement. The length must be `C'`.
        """
        sample = S2THubInterface.get_model_input(self.task, inputs)
        (text, (wav, sr)) = S2THubInterface.get_prediction(
            self.task, self.model, self.generator, sample, synthesize_speech=True
        )
        return wav.numpy(), sr, [text]
