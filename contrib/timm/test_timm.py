import timm

from ..utils import production_endpoint


MODEL_ID = "nateraw/timm-resnet50-beans"


def test_load_and_push_to_hub(repo_name: str, cleanup_repo: None) -> None:
    # Test load only config
    with production_endpoint():
        _ = timm.models.hub.load_model_config_from_hf(MODEL_ID)

    # Load entire model from Hub
    with production_endpoint():
        model = timm.create_model("hf_hub:" + MODEL_ID, pretrained=True)

    # Push model to Hub
    timm.models.hub.push_to_hf_hub(model, repo_name)
