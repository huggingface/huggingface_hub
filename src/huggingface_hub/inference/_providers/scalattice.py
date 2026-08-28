from ._common import BaseConversationalTask


class ScalatticeConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="scalattice", base_url="https://api.scalattice.cloud")
