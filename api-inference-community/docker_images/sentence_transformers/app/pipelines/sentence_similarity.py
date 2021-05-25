from typing import Dict, List, Union

from app.pipelines import Pipeline
from sentence_transformers import SentenceTransformer, util


class SentenceSimilarityPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
    ):
        self.model = SentenceTransformer(model_id)

    def __call__(self, inputs: Dict[str, Union[str, List[str]]]) -> List[float]:
        """
        Args:
            inputs (:obj:`dict`):
                a dictionary containing two keys, 'source_sentence' mapping
                to the sentence that will be compared against all the others,
                and 'sentences', mapping to a list of strings to which the
                source will be compared.
        Return:
            A :obj:`list` of floats: Cosine similarity between `source_sentence` and each sentence from `sentences`.
        """
        embeddings1 = self.model.encode(
            inputs["source_sentence"], convert_to_tensor=True
        )
        embeddings2 = self.model.encode(inputs["sentences"], convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(embeddings1, embeddings2).tolist()[0]
        return similarities
