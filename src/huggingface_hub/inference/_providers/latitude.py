from ._common import BaseConversationalTask, BaseTextGenerationTask


class LatitudeConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="latitude-sh", base_url="https://api.lsh.ai")


class LatitudeTextGenerationTask(BaseTextGenerationTask):
    def __init__(self):
        super().__init__(provider="latitude-sh", base_url="https://api.lsh.ai")
