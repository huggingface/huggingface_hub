from typing import Dict, List


# Must contain at least one example of each implemented pipeline
# Tests do not check the actual values of the model output, so small dummy
# models are recommended for faster tests.
TESTABLE_MODELS: Dict[str, List[str]] = {
    "audio-to-audio": ["osanseviero/ConvTasNet_Libri1Mix_enhsingle_16k"],
    "automatic-speech-recognition": ["osanseviero/pyctcdecode_asr"],
    # This is very slow the first time as fasttext model is large.
    "feature-extraction": ["osanseviero/fasttext_english"],
    "image-classification": ["osanseviero/fastai_cat_vs_dog"],
    "structured-data-classification": ["osanseviero/wine-quality"],
    "text-classification": ["osanseviero/fasttext_nearest"],
    "text-to-image": ["osanseviero/BigGAN-deep-128"],
    "token-classification": ["osanseviero/en_core_web_sm"],
}
