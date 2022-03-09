from typing import Any, Dict, List

from app.pipelines import Pipeline

from PIL import Image
import torch
from torchvision.transforms import Compose, ConvertImageDtype, PILToTensor
from doctr.models.obj_detection.factory import from_hub


class ObjectDetectionPipeline(Pipeline):
    def __init__(self, model_id: str):

        self.model = from_hub(model_id).eval()

        self.transform = Compose(
            [
                PILToTensor(),
                ConvertImageDtype(torch.float32),
            ]
        )

        self.labels = self.model.cfg.get("classes")
        if self.labels is None:
            self.labels = [f"LABEL_{i}" for i in range(self.model.num_classes)]

    def __call__(self, inputs: Image.Image) -> List[Dict[str, Any]]:
        """
        Args:
            inputs (:obj:`PIL.Image`):
                The raw image representation as PIL.
                No transformation made whatsoever from the input. Make all necessary transformations here.
        Return:
            A :obj:`list`:. The list contains items that are dicts with the keys "label", "score" and "box".
        """
        im = inputs.convert("RGB")
        inputs = self.transform(im).unsqueeze(0)

        with torch.inference_mode():
            out = self.model(inputs)[0]

        return [
            {
                "label": self.labels[idx],
                "score": score.item(),
                "box": {
                    "xmin": int(round(box[0].item())),
                    "ymin": int(round(box[1].item())),
                    "xmax": int(round(box[2].item())),
                    "ymax": int(round(box[3].item())),
                },
            }
            for idx, score, box in zip(out["labels"], out["scores"], out["boxes"])
        ]
