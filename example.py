import torch
from huggingface_hub.file_download import hf_hub_download

OLDER_REVISION = "bbc77c8132af1cc5cf678da3f1ddf2de43606d48"

hf_hub_download("julien-c/EsperBERTo-small", filename="README.md")

hf_hub_download("julien-c/EsperBERTo-small", filename="pytorch_model.bin")

hf_hub_download(
    "julien-c/EsperBERTo-small", filename="README.md", revision=OLDER_REVISION
)

weights_file = hf_hub_download(
    "julien-c/EsperBERTo-small", filename="pytorch_model.bin", revision=OLDER_REVISION
)

w = torch.load(weights_file, map_location=torch.device("cpu"))
# Yay it works! just loaded a torch file from a symlink

print()
