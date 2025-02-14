from ._common import BaseTextGenerationTask


class FireworksAIConversationalTask(BaseTextGenerationTask):
    def __init__(self):
        super().__init__(provider="fireworks-ai", base_url="https://api.fireworks.ai/inference", task="conversational")
