from ._common import BaseConversationalTask


class IntelCSAIConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="intelcs-ai-iaas", base_url="https://iaas-router.iaas-billing.workers.dev")
