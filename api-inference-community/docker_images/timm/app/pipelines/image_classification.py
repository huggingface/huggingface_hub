from typing import Any, Dict, List

import timm
import torch
from app.pipelines import Pipeline
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models.hub import load_model_config_from_hf


class ImageClassificationPipeline(Pipeline):
    def __init__(self, model_id: str):

        self.hf_cfg, self.arch = load_model_config_from_hf(model_id)
        self.model = timm.create_model(f"hf_hub:{model_id}", pretrained=True)
        self.transform = create_transform(
            **resolve_data_config(self.hf_cfg, model=self.model)
        )
        self.model.eval()

        self.top_k = min(self.model.num_classes, 5)

        self.labels = self.hf_cfg.get("labels", None)
        if self.labels is None:
            self.labels = [f"LABEL_{i}" for i in range(self.model.num_classes)]

    def __call__(self, inputs: Image.Image) -> List[Dict[str, Any]]:
        """
        Args:
            inputs (:obj:`PIL.Image`):
                The raw image representation as PIL.
                No transformation made whatsoever from the input. Make all necessary transformations here.
        Return:
            A :obj:`list`:. The list contains items that are dicts should be liked {"label": "XXX", "score": 0.82}
                It is preferred if the returned list is in decreasing `score` order
        """
        im = inputs.convert("RGB")
        inputs = self.transform(im).unsqueeze(0)

        with torch.no_grad():
            out = self.model(inputs)

        probabilities = torch.nn.functional.softmax(out[0], dim=0)

        values, indices = torch.topk(probabilities, self.top_k)

        labels = [
            {"label": self.labels[i], "score": v.item()}
            for i, v in zip(indices, values)
        ]
        return labels
