from ._common import BaseInferenceTask


class Summarization(BaseInferenceTask):
    TASK_NAME = "summarization"


build_url = Summarization.build_url
map_model = Summarization.map_model
prepare_headers = Summarization.prepare_headers
prepare_payload = Summarization.prepare_payload
get_response = Summarization.get_response
