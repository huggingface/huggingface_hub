import numpy as np
from app.pipelines import Pipeline


class AudioSourceSeparationPipeline(Pipeline):
    def __init__(self, model_id: str):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        raise NotImplementedError(
            "Please implement AudioSourceSeparationPipeline __init__ function"
        )

    def __call__(self, inputs: np.array) -> bytes:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default at 16KHz.
                Check `app.validation` if a different sample rate is required
                or if it depends on the model
        Return:
            A :obj:`bytes`:. The raw audio waveform as bytes encode as a wav file.
        """
        # IMPLEMENT_THIS
        raise NotImplementedError(
            "Please implement AudioSourceSeparationPipeline __call__ function"
        )
