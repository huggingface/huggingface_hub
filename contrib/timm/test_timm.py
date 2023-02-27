import timm

from ..utils import production_endpoint


MODEL_ID = "timm/mobilenetv3_large_100.ra_in1k"


@production_endpoint()
def test_load_from_hub() -> None:
    # Test load only config
    _ = timm.models.load_model_config_from_hf(MODEL_ID)

    # Load entire model from Hub
    _ = timm.create_model("hf_hub:" + MODEL_ID, pretrained=True)


def test_push_to_hub(repo_name: str, cleanup_repo: None) -> None:
    model = timm.create_model("mobilenetv3_rw")
    timm.models.push_to_hf_hub(model, repo_name)
