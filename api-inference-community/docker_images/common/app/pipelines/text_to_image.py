from app.pipelines import Pipeline


class TextToImagePipeline(Pipeline):
    def __init__(self, model_id: str):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need for inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        raise NotImplementedError(
            "Please implement TextToImagePipeline.__init__ function"
        )

    def __call__(self, inputs: str) -> str:
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`str`. A base64 string representing the image.
        """
        # IMPLEMENT_THIS
        raise NotImplementedError(
            "Please implement TextToImagePipeline __call__ function"
        )
