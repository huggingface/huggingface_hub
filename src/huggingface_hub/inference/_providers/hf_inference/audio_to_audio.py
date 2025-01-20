from ._common import BaseInferenceTask


class AudioToAudio(BaseInferenceTask):
    TASK_NAME = "audio-to-audio"


build_url = AudioToAudio.build_url
map_model = AudioToAudio.map_model
prepare_headers = AudioToAudio.prepare_headers
prepare_payload = AudioToAudio.prepare_payload
get_response = AudioToAudio.get_response
