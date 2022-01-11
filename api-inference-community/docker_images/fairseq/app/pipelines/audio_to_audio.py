import os
from typing import List, Tuple

import numpy as np
import torch
from app.pipelines import Pipeline
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface


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

        self.sampling_rate = getattr(self.task, "sr", None) or 16_000

        tgt_lang = self.task.data_cfg.hub.get("tgt_lang", None)
        pfx = f"{tgt_lang}_" if self.task.data_cfg.prepend_tgt_lang_tag else ""
        tts_model_id = self.task.data_cfg.hub.get(f"{pfx}tts_model_id", None)
        self.tts_model, self.tts_task, self.tts_generator = None, None, None
        if tts_model_id is not None:
            _repo, _id = tts_model_id.split(":")
            (
                tts_models,
                tts_cfg,
                self.tts_task,
            ) = load_model_ensemble_and_task_from_hf_hub(
                f"facebook/{_id}",
                arg_overrides={"vocoder": "griffin_lim", "fp16": False},
                cache_dir=os.getenv("HUGGINGFACE_HUB_CACHE"),
            )
            self.tts_model = tts_models[0].cpu()
            self.tts_model.eval()
            tts_cfg["task"].cpu = True
            TTSHubInterface.update_cfg_with_data_cfg(tts_cfg, self.tts_task.data_cfg)
            self.tts_generator = self.tts_task.build_generator(
                [self.tts_model], tts_cfg
            )

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
        _inputs = torch.from_numpy(inputs).unsqueeze(0)
        sample = S2THubInterface.get_model_input(self.task, _inputs)
        text = S2THubInterface.get_prediction(
            self.task, self.model, self.generator, sample
        )

        if self.tts_model is None:
            return np.zeros((0,)), self.sampling_rate, [text]
        else:
            tts_sample = TTSHubInterface.get_model_input(self.tts_task, text)
            wav, sr = TTSHubInterface.get_prediction(
                self.tts_task, self.tts_model, self.tts_generator, tts_sample
            )
            return wav.unsqueeze(0).numpy(), sr, [text]
