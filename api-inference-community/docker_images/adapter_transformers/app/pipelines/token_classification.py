from typing import Any, Dict, List

from app.pipelines import Pipeline
from transformers import AutoModelWithHeads, AutoTokenizer, get_adapter_info
from transformers import TokenClassificationPipeline as TransformersTokenClassificationPipeline


class TokenClassificationPipeline(Pipeline):
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
        self.pipeline = TransformersTokenClassificationPipeline(model, tokenizer)

    def __call__(self, inputs: str) -> List[Dict[str, Any]]:
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`list`:. The object returned should be like [{"entity_group": "XXX", "word": "some word", "start": 3, "end": 6, "score": 0.82}] containing :
                - "entity_group": A string representing what the entity is.
                - "word": A rubstring of the original string that was detected as an entity.
                - "start": the offset within `input` leading to `answer`. context[start:stop] == word
                - "end": the ending offset within `input` leading to `answer`. context[start:stop] === word
                - "score": A score between 0 and 1 describing how confident the model is for this entity.
        """
        self.pipeline(inputs)
