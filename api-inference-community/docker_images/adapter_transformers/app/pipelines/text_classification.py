from typing import Dict, List

from app.pipelines import Pipeline
from transformers import (
    TextClassificationPipeline as TransformersClassificationPipeline,
)


class TextClassificationPipeline(Pipeline):
    def __init__(
        self,
        adapter_id: str,
    ):
        self.pipeline = self._load_pipeline_instance(
            TransformersClassificationPipeline, adapter_id
        )

    def __call__(self, inputs: str) -> List[Dict[str, float]]:
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`list`:. The object returned should be like [{"label": 0.9939950108528137}] containing :
                - "label": A string representing what the label/class is. There can be multiple labels.
                - "score": A score between 0 and 1 describing how confident the model is for this label/class.
        """
        try:
            return [self.pipeline(inputs, return_all_scores=True)[0]]
        except Exception as e:
            raise ValueError(e)
