import os
from typing import Tuple

import numpy as np
from app.pipelines import Pipeline
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface


class TextToSpeechPipeline(Pipeline):
    def __init__(self, model_id: str):
        model, cfg, task = load_model_ensemble_and_task_from_hf_hub(
            model_id,
            arg_overrides={"vocoder": "griffin_lim", "fp16": False},
            cache_dir=os.getenv("HUGGINGFACE_HUB_CACHE"),
        )
        self.model = model[0].cpu()
        self.model.eval()
        cfg["task"].cpu = True
        self.task = task
        TTSHubInterface.update_cfg_with_data_cfg(cfg, self.task.data_cfg)
        self.generator = self.task.build_generator(model, cfg)

    def __call__(self, inputs: str) -> Tuple[np.array, int]:
        """
        Args:
            inputs (:obj:`str`):
                The text to generate audio from
        Return:
            A :obj:`np.array` and a :obj:`int`: The raw waveform as a numpy
            array, and the sampling rate as an int.
        """
        inputs = inputs.strip("\x00")
        if len(inputs) == 0:
            return np.zeros((0,)), self.task.sr

        sample = TTSHubInterface.get_model_input(self.task, inputs)
        wav, sr = TTSHubInterface.get_prediction(
            self.task, self.model, self.generator, sample
        )
        return wav.numpy(), sr
