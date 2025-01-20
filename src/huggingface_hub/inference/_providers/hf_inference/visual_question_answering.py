from ._common import BaseInferenceTask


class VisualQuestionAnswering(BaseInferenceTask):
    TASK_NAME = "visual-question-answering"


build_url = VisualQuestionAnswering.build_url
map_model = VisualQuestionAnswering.map_model
prepare_headers = VisualQuestionAnswering.prepare_headers
prepare_payload = VisualQuestionAnswering.prepare_payload
get_response = VisualQuestionAnswering.get_response
