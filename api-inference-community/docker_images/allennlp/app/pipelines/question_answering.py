import os
import shutil
from typing import Any, Dict

# Even though it is not imported, it is actually required, it downlaods some stuff.
import allennlp_models  # noqa: F401
from allennlp.predictors.predictor import Predictor
from app.pipelines import Pipeline


class QuestionAnsweringPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
    ):
        try:
            self.predictor = Predictor.from_path("hf://" + model_id)
        except (IOError, OSError):
            nltk = os.getenv("NLTK_DATA")
            if nltk is None:
                raise
            directory = os.path.join(nltk, "corpora")
            shutil.rmtree(directory)
            self.predictor = Predictor.from_path("hf://" + model_id)

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
        allenlp_input = {"passage": inputs["context"], "question": inputs["question"]}
        predictions = self.predictor.predict_json(allenlp_input)

        start_token_idx, end_token_idx = predictions["best_span"]
        start = predictions["token_offsets"][start_token_idx][0]
        end = predictions["token_offsets"][end_token_idx][1]

        score = (
            predictions["span_end_probs"][end_token_idx]
            * predictions["span_start_probs"][start_token_idx]
        )

        return {
            "answer": predictions["best_span_str"],
            "start": start,
            "end": end,
            "score": score,
        }
