from typing import Tuple

import numpy as np
from app.pipelines import Pipeline
from speechbrain.pretrained import SepformerSeparation
import torch
import torchaudio


class AudioSourceSeparationPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model = SepformerSeparation.from_hparams(source=model_id)

        
    def __call__(self, inputs: np.array) -> Tuple[np.array, int]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default at 16KHz.
                Check `app.validation` if a different sample rate is required
                or if it depends on the model
        Return:
            est_sources: np.array  
            
                The raw waveforms as a numpy array of size T x S, where T is number of time points and S is the number of sources, and the sampling rate as an int.

            fs : int:
                The sampling frequency for the estimated sources
        """


        fs_model = self.model.hparams.sample_rate

        # we resample to the model's sampling frequency if needed
        # we assume that the input is at 16kHz
        mix = torch.from_numpy(inputs)
        if 16000 != fs_model:
            resamp = torchaudio.transforms.Resample(
                orig_freq=16000, new_freq=fs_model
            )
            mix = resamp(mix)

        # separate
        est_sources = self.model.separate_batch(mix)

        # normalize for loudness
        est_sources = est_sources / est_sources.max(dim=1, keepdim=True)[0]

        # return estimated sources as a T(time) x S(n.sources) matrix 
        return est_sources[0].numpy(), fs_model

         

        
        
