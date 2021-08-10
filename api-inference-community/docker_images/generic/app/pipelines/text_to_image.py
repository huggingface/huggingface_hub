from typing import TYPE_CHECKING

from app.pipelines import Pipeline

if TYPE_CHECKING:
    from PIL import Image

class TextToImagePipeline(Pipeline):
    def __init__(self, model_id: str):
        super().__init__(model_id)

    def __call__(self, inputs: str) -> "Image.Image":
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`PIL.Image` with the raw image representation as PIL.
        """
        return super().__call__(inputs)
