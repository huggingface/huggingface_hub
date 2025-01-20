from ._common import BaseInferenceTask


class FillMask(BaseInferenceTask):
    TASK_NAME = "fill-mask"


build_url = FillMask.build_url
map_model = FillMask.map_model
prepare_headers = FillMask.prepare_headers
prepare_payload = FillMask.prepare_payload
get_response = FillMask.get_response
