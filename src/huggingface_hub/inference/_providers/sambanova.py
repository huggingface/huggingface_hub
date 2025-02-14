from huggingface_hub.inference._providers._common import BaseTextGenerationTask


class SambanovaConversationalTask(BaseTextGenerationTask):
    def __init__(self):
        super().__init__(provider="sambanova", base_url="https://api.sambanova.ai", task="conversational")
