from huggingface_hub.snapshot_download import snapshot_download
from huggingface_hub.utils.logging import set_verbosity_debug
from transformers import AutoModelForMaskedLM, TFAutoModelForMaskedLM


set_verbosity_debug()

DISTILBERT = "distilbert-base-uncased"

folder_path = snapshot_download(
    repo_id=DISTILBERT,
    repo_type="model",
)


print("The whole model repo has been saved to", folder_path)

pt_model = AutoModelForMaskedLM.from_pretrained(folder_path)
tf_model = TFAutoModelForMaskedLM.from_pretrained(folder_path)
# Yay it works!

print()
