from ._common import BaseInferenceTask


class ImageSegmentation(BaseInferenceTask):
    TASK_NAME = "image-segmentation"


build_url = ImageSegmentation.build_url
map_model = ImageSegmentation.map_model
prepare_headers = ImageSegmentation.prepare_headers
prepare_payload = ImageSegmentation.prepare_payload
get_response = ImageSegmentation.get_response
