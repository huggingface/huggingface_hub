from ._common import BaseInferenceTask


class AutomaticSpeechRecognition(BaseInferenceTask):
    TASK_NAME = "automatic-speech-recognition"


build_url = AutomaticSpeechRecognition.build_url
map_model = AutomaticSpeechRecognition.map_model
prepare_headers = AutomaticSpeechRecognition.prepare_headers
prepare_payload = AutomaticSpeechRecognition.prepare_payload
get_response = AutomaticSpeechRecognition.get_response
