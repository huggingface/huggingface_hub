from typing import Generator

import pytest

from huggingface_hub import HfFolder


@pytest.fixture(autouse=True, scope="session")
def clean_hf_folder_token_for_tests() -> Generator:
    """Clean token stored on machine before all tests and reset it back at the end.

    Useful to avoid token deletion when running tests locally.
    """
    # Remove registered token
    token = HfFolder().get_token()
    HfFolder().delete_token()

    yield  # Run all tests

    # Set back token once all tests have passed
    if token is not None:
        HfFolder().save_token(token)
