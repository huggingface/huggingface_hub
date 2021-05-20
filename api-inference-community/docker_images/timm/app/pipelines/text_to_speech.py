from app.pipelines import Pipeline


class TextToSpeechPipeline(Pipeline):
    def __init__(self, model_id: str):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        raise NotImplementedError(
            "Please implement TextToSpeechPipeline __init__ function"
        )

    def __call__(self, inputs: str) -> bytes:
        """
        Args:
            inputs (:obj:`str`):
                The text to generate audio from
        Return:
            A :obj:`bytes`:. The raw audio encoded as a wav format.
        """
        # IMPLEMENT_THIS
        raise NotImplementedError(
            "Please implement TextToSpeechPipeline __call__ function"
        )
