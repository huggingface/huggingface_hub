from downstream.runner import Runner

class PreTrainedModel():
    def __init__(self):
        """
        Loads model and tokenizer from local directory
        """
        torch.load("hurbert_base_asr.ckpt", map_location='cpu')
        config = 
        
        self.model = AutomaticSpeechRecognitionPipeline(model=model, feature_extractor=extractor, tokenizer=tokenizer)
    
        
    def __call__(self, inputs)-> Dict[str, str]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default at 16KHz.
        Return:
            A :obj:`dict`:. The object return should be liked {"text": "XXX"} containing
            the detected text from the input audio.
        """
        return self.model(inputs)