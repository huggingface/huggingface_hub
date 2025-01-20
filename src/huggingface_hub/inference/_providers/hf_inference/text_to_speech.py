from ._common import BaseInferenceTask


class TextToSpeech(BaseInferenceTask):
    TASK_NAME = "text-to-speech"


build_url = TextToSpeech.build_url
map_model = TextToSpeech.map_model
prepare_headers = TextToSpeech.prepare_headers
prepare_payload = TextToSpeech.prepare_payload
get_response = TextToSpeech.get_response
