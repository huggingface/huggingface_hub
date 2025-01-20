from ._common import BaseInferenceTask


class ImageToImage(BaseInferenceTask):
    TASK_NAME = "image-to-image"


build_url = ImageToImage.build_url
map_model = ImageToImage.map_model
prepare_headers = ImageToImage.prepare_headers
prepare_payload = ImageToImage.prepare_payload
get_response = ImageToImage.get_response
