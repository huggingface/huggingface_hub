from typing import Tuple

import numpy as np
from app.pipelines import Pipeline
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf
import torch
import g2p_en


class TextToSpeechPipeline(Pipeline):
    def __init__(self, model_id: str):
        # hardcoded stuff can later be moved to a config
        model, cfg, task = load_model_ensemble_and_task_from_hf(model_id, arg_overrides={"vocoder": "griffin_lim", "fp16": False}, device="cpu")

        self.task = task
        self.generator = task.build_generator(model, cfg)
        self.model = model[0]

        # 16000 by default if not specified
        self.sampling_rate = 16000

    def _tokenize(self, text):
        tokenized = g2p_en.G2p()(text)
        tokenized = [{",": "sp", ";": "sp"}.get(p, p) for p in tokenized]
        return " ".join(p for p in tokenized if p.isalnum())

    def __call__(self, inputs: str) -> Tuple[np.array, int]:
        """
        Args:
            inputs (:obj:`str`):
                The text to generate audio from
        Return:
            A :obj:`np.array` and a :obj:`int`: The raw waveform as a numpy array, and the sampling rate as an int.
        """
        tokenized_inputs = self._tokenize(inputs)

        sample = {
            "net_input": {
                "src_tokens": self.task.src_dict.encode_line(tokenized_inputs).view(1, -1),
                "src_lengths": torch.Tensor([len(tokenized_inputs.split())]).long(),
                "prev_output_tokens": None
            },
            "target_lengths": None,
            "speaker": None,
        }

        with torch.no_grad():
            generation = self.generator.generate(self.model, sample)

        waveform = generation[0]["waveform"].cpu().numpy()

        return waveform, self.sampling_rate
