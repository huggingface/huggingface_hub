from ._common import BaseInferenceTask


class AudioClassification(BaseInferenceTask):
    TASK_NAME = "audio-classification"


build_url = AudioClassification.build_url
map_model = AudioClassification.map_model
prepare_headers = AudioClassification.prepare_headers
prepare_payload = AudioClassification.prepare_payload
get_response = AudioClassification.get_response
