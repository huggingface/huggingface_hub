from typing import Dict, List

import numpy as np
import torch
from app.pipelines import Pipeline
from speechbrain.pretrained import EncoderClassifier


class AudioClassificationPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model = EncoderClassifier.from_hparams(source=model_id)

        self.top_k = 5

        # Please define a `self.sampling_rate` for this pipeline
        # to automatically read the input correctly
        self.sampling_rate = 16000

    def __call__(self, inputs: np.array) -> List[Dict[str, float]]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default at 16KHz.
        Return:
            A :obj:`list`:. The object returned should be a list like [{"label": "text", "score": 0.9939950108528137}] containing :
                - "label": A string representing what the label/class is. There can be multiple labels.
                - "score": A score between 0 and 1 describing how confident the model is for this label/class.
        """
        batch = torch.from_numpy(inputs).unsqueeze(0)
        rel_length = torch.tensor([1.0])
        probs, _, _, _ = self.model.classify_batch(batch, rel_length)
        probs = torch.softmax(probs[0], dim=0)
        labels = self.model.hparams.label_encoder.decode_ndim(range(len(probs)))
        results = []
        for label, prob in sorted(zip(labels, probs), reverse=True)[: self.top_k]:
            results.append({"label": label, "score": prob.item()})
        return results
