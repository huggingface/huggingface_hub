import contextlib
from typing import Generator
from unittest.mock import patch


@contextlib.contextmanager
def production_endpoint() -> Generator:
    """Patch huggingface_hub to connect to production server in a context manager.

    Ugly way to patch all constants at once.
    TODO: refactor when https://github.com/huggingface/huggingface_hub/issues/1172 is fixed.

    Example:
    ```py
    def test_push_to_hub():
        # Pull from production Hub
        with production_endpoint():
            model = ...from_pretrained("modelname")

        # Push to staging Hub
        model.push_to_hub()
    ```
    """
    PROD_ENDPOINT = "https://huggingface.co"
    ENDPOINT_TARGETS = [
        "huggingface_hub.constants",
        "huggingface_hub._commit_api",
        "huggingface_hub.hf_api",
        "huggingface_hub.lfs",
        "huggingface_hub.commands.user",
        "huggingface_hub.utils._git_credential",
    ]

    PROD_URL_TEMPLATE = PROD_ENDPOINT + "/{repo_id}/resolve/{revision}/{filename}"
    URL_TEMPLATE_TARGETS = [
        "huggingface_hub.constants",
        "huggingface_hub.file_download",
    ]

    from huggingface_hub.hf_api import api

    patchers = (
        [patch(target + ".ENDPOINT", PROD_ENDPOINT) for target in ENDPOINT_TARGETS]
        + [
            patch(target + ".HUGGINGFACE_CO_URL_TEMPLATE", PROD_URL_TEMPLATE)
            for target in URL_TEMPLATE_TARGETS
        ]
        + [patch.object(api, "endpoint", PROD_URL_TEMPLATE)]
    )

    # Start all patches
    for patcher in patchers:
        patcher.start()

    yield

    # Stop all patches
    for patcher in patchers:
        patcher.stop()
