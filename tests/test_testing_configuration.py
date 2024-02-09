from huggingface_hub import get_token


def test_no_token_in_staging_environment():
    """Make sure no token is set in test environment."""
    assert get_token() is None
