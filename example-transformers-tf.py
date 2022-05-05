from huggingface_hub.snapshot_download import snapshot_download
from huggingface_hub.utils.logging import set_verbosity_debug

set_verbosity_debug()

DISTILBERT = "distilbert-base-uncased"

folder_path = snapshot_download(
    repo_id=DISTILBERT,
    repo_type="model",
)

print("loading TF model from", folder_path)


print()
