from typing import Any, Dict

from app.pipelines import Pipeline


class QuestionAnsweringPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
    ):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        super().__init__(model_id)

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
        return super().__call__(inputs)
