# TO BE DELETED
import asyncio

from huggingface_hub import snapshot_download


"z-uo/male-LJSpeech-italian"
"oscar"

asyncio.run(
    snapshot_download(
        repo_id="z-uo/male-LJSpeech-italian",
        repo_type="dataset",
    )
)
