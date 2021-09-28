from typing import TYPE_CHECKING, Any, Dict, List

from app.pipelines import Pipeline


if TYPE_CHECKING:
    from PIL import Image


class ImageClassificationPipeline(Pipeline):
    def __init__(self, model_id: str):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        raise NotImplementedError(
            "Please implement ImageClassificationPipeline __init__ function"
        )

    def __call__(self, inputs: "Image.Image") -> List[Dict[str, Any]]:
        """
        Args:
            inputs (:obj:`PIL.Image`):
                The raw image representation as PIL.
                No transformation made whatsoever from the input. Make all necessary transformations here.
        Return:
            A :obj:`list`:. The list contains items that are dicts should be liked {"label": "XXX", "score": 0.82}
                It is preferred if the returned list is in decreasing `score` order
        """
        # IMPLEMENT_THIS
        raise NotImplementedError(
            "Please implement ImageClassificationPipeline __call__ function"
        )
