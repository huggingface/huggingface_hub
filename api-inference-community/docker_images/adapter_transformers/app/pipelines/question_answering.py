from typing import Any, Dict

from app.pipelines import Pipeline
from transformers import AutoModelWithHeads, AutoTokenizer, get_adapter_info
from transformers import QuestionAnsweringPipeline as TransformersQAPipeline


class QuestionAnsweringPipeline(Pipeline):
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
        self.pipeline = TransformersQAPipeline(model, tokenizer)

    def __call__(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """
        Args:
            inputs (:obj:`dict`):
                a dictionnary containing two keys, 'question' being the question being asked and 'context' being some text containing the answer.
        Return:
            A :obj:`dict`:. The object return should be like {"answer": "XXX", "start": 3, "end": 6, "score": 0.82} containing :
                - "answer": the extracted answer from the `context`.
                - "start": the offset within `context` leading to `answer`. context[start:stop] == answer
                - "end": the ending offset within `context` leading to `answer`. context[start:stop] === answer
                - "score": A score between 0 and 1 describing how confident the model is for this answer.
        """
        return self.pipeline(**inputs)
