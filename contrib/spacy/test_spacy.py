import time

from spacy_huggingface_hub import push

from huggingface_hub import delete_repo, hf_hub_download, model_info
from huggingface_hub.errors import HfHubHTTPError

from ..utils import production_endpoint


def test_push_to_hub(user: str) -> None:
    """Test equivalent of `python -m spacy huggingface-hub push`.

    (0. Delete existing repo on the Hub (if any))
    1. Download an example file from production
    2. Push the model!
    3. Check model pushed the Hub + as spacy library
    (4. Cleanup)
    """
    model_id = f"{user}/en_core_web_sm"
    _delete_repo(model_id)

    # Download example file from HF Hub (see https://huggingface.co/spacy/en_core_web_sm)
    with production_endpoint():
        whl_path = hf_hub_download(
            repo_id="spacy/en_core_web_sm",
            filename="en_core_web_sm-any-py3-none-any.whl",
        )

    # Push spacy model to Hub
    push(whl_path)

    # Sleep to ensure that model_info isn't called too soon
    time.sleep(1)

    # Check model has been pushed properly
    model_id = f"{user}/en_core_web_sm"
    assert model_info(model_id).library_name == "spacy"

    # Cleanup
    _delete_repo(model_id)


def _delete_repo(model_id: str) -> None:
    try:
        delete_repo(model_id)
    except HfHubHTTPError:
        pass
