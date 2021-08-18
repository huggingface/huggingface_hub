from typing import Dict, List

from app.pipelines import Pipeline
from transformers import AutoModelWithHeads, AutoTokenizer, get_adapter_info
from transformers import TextClassificationPipeline as TransformersClassificationPipeline


class TextClassificationPipeline(Pipeline):
    def __init__(
        self,
        adapter_id: str,
    ):
        adapter_info = get_adapter_info(adapter_id, source="hf")
        if adapter_info is None:
            raise ValueError(f"Adapter with id '{adapter_id}' not available.")

        tokenizer = AutoTokenizer.from_pretrained(adapter_info.model_name)
        model = AutoModelWithHeads.from_pretrained(adapter_info.model_name)
        model.load_adapter(adapter_id, source="hf", set_active=True)
        self.pipeline = TransformersClassificationPipeline(model=model, tokenizer=tokenizer)

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
        return self.pipeline(inputs)
