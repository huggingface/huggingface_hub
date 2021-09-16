from enum import Enum

from huggingface_hub import HfApi


class ModelType(Enum):
    # audio-to-audio
    SEPFORMERSEPARATION = "SEPFORMERSEPARATION"
    SPECTRALMASKENHANCEMENT = "SPECTRALMASKENHANCEMENT"
    # automatic-speech-recognition
    ENCODERASR = "ENCODERASR"
    ENCODERDECODERASR = "ENCODERDECODERASR"
    # audio-clasification
    ENCODERCLASSIFIER = "ENCODERCLASSIFIER"


def get_type(model_id):
    info = HfApi().model_info(repo_id=model_id)
    if info.config:
        if "speechbrain" in info.config:
            return ModelType(info.config["speechbrain"]["interface"].upper())
        else:
            raise ValueError("speechbrain not in config.json")
    raise ValueError("no config.json in repository")
