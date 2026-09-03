from ._common import BaseConversationalTask

# Partner origin Hugging Face registers server-side. BaseConversationalTask
# appends `/v1/chat/completions`, so this must be the origin without a `/v1`
# suffix. HF-token traffic is rewritten to
# `https://router.huggingface.co/neuronpool` by TaskProviderHelper.
NEURONPOOL_API_BASE_URL = "https://api.neuronpool.dev"


class NeuronpoolConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="neuronpool", base_url=NEURONPOOL_API_BASE_URL)
