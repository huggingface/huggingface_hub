from app.pipelines import Pipeline


class TextToImagePipeline(Pipeline):
    def __init__(self, model_id: str):
        super().__init__(model_id)

    def __call__(self, inputs: str) -> str:
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`str`. A base64 string representing the image.
        """
        return super().__call__(inputs)
