from ._common import BaseInferenceTask


class TableQuestionAnswering(BaseInferenceTask):
    TASK_NAME = "table-question-answering"


build_url = TableQuestionAnswering.build_url
map_model = TableQuestionAnswering.map_model
prepare_headers = TableQuestionAnswering.prepare_headers
prepare_payload = TableQuestionAnswering.prepare_payload
get_response = TableQuestionAnswering.get_response
