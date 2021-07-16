from typing import Dict, List, Union

from app.pipelines import Pipeline


class SentenceSimilarityPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
    ):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        raise NotImplementedError(
            "Please implement SentenceSimilarityPipeline __init__ function"
        )

    def __call__(self, inputs: Dict[str, Union[str, List[str]]]) -> List[float]:
        """
        Args:
            inputs (:obj:`dict`):
                a dictionary containing two keys, 'source_sentence' mapping
                to the sentence that will be compared against all the others,
                and 'sentences', mapping to a list of strings to which the
                source will be compared.
        Return:
            A :obj:`list` of floats: Some similarity measure between `source_sentence` and each sentence from `sentences`.
        """
        # IMPLEMENT_THIS
        raise NotImplementedError(
            "Please implement SentenceSimilarityPipeline __call__ function"
        )
