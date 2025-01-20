from ._common import BaseInferenceTask


class QuestionAnswering(BaseInferenceTask):
    TASK_NAME = "question-answering"


build_url = QuestionAnswering.build_url
map_model = QuestionAnswering.map_model
prepare_headers = QuestionAnswering.prepare_headers
prepare_payload = QuestionAnswering.prepare_payload
get_response = QuestionAnswering.get_response
