from ._common import BaseInferenceTask


class Translation(BaseInferenceTask):
    TASK_NAME = "translation"


build_url = Translation.build_url
map_model = Translation.map_model
prepare_headers = Translation.prepare_headers
prepare_payload = Translation.prepare_payload
get_response = Translation.get_response
