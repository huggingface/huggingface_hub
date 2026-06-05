from typing import Any, Optional, Union
from ._common import BaseTextGenerationTask

class MyCloudInfraTextGenerationTask(BaseTextGenerationTask):
    def __init__(self):
        # 这里的 provider 标识符要跟我们在 hub-docs 里的配置完全一致："my-cloud-infra"
        super().__init__(
            provider="my-cloud-infra",
            base_url="https://calibration-gentle-constantly-liked.trycloudflare.com",
            task="text-generation"
        )
