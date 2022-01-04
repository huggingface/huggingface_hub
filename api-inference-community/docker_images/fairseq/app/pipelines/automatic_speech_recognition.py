import os
from typing import Dict

import numpy as np
import torch
from app.pipelines import Pipeline
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.speech_to_text.hub_interface import S2THubInterface


class SpeechToTextPipeline(Pipeline):
    def __init__(self, model_id: str):
        model, cfg, task = load_model_ensemble_and_task_from_hf_hub(
            model_id,
            arg_overrides={"config_yaml": "config.yaml"},
            cache_dir=os.getenv("HUGGINGFACE_HUB_CACHE"),
        )
        self.model = model[0].cpu()
        self.model.eval()
        cfg["task"].cpu = True
        self.task = task
        self.generator = task.build_generator(self.model, cfg)

    def __call__(self, inputs: np.array) -> Dict[str, str]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default at 16KHz.
                Check `app.validation` if a different sample rate is required
                or if it depends on the model
        Return:
            A :obj:`dict`:. The object return should be liked {"text": "XXX"}
            containing the detected language from the input audio
        """
        _inputs = torch.from_numpy(inputs).unsqueeze(0)  # T -> 1 x T
        sample = S2THubInterface.get_model_input(self.task, inputs)
        pred = S2THubInterface.get_prediction(
            self.task, self.model, self.generator, sample
        )
        return {"text": pred}
