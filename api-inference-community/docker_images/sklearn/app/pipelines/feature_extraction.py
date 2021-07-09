from typing import List

from app.pipelines import Pipeline


class FeatureExtractionPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
    ):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        raise NotImplementedError(
            "Please implement FeatureExtractionPipeline __init__ function"
        )

    def __call__(self, inputs: str) -> List[float]:
        """
        Args:
            inputs (:obj:`str`):
                a string to get the features of.
        Return:
            A :obj:`list` of floats: The features computed by the model.
        """
        # IMPLEMENT_THIS
        raise NotImplementedError(
            "Please implement FeatureExtractionPipeline __call__ function"
        )
