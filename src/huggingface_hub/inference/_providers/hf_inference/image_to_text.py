from ._common import BaseInferenceTask


class ImageToText(BaseInferenceTask):
    TASK_NAME = "image-to-text"


build_url = ImageToText.build_url
map_model = ImageToText.map_model
prepare_headers = ImageToText.prepare_headers
prepare_payload = ImageToText.prepare_payload
get_response = ImageToText.get_response
