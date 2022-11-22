import timm


def test_push_to_hub(repo_name: str, cleanup_repo: None) -> None:
    # Build a model 🔧
    model = timm.create_model("resnet18", pretrained=True, num_classes=4)

    # Push it to the 🤗 hub
    timm.models.hub.push_to_hf_hub(
        model, repo_name, model_config=dict(labels=["a", "b", "c", "d"])
    )
