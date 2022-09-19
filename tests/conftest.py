import pytest

from huggingface_hub.utils._headers import _is_private, _is_valid_token


@pytest.fixture(autouse=True)
def disable_lru_cache_in_all_tests() -> None:
    """Clean lru_cache between each test to ensure independence between them."""
    _is_private.cache_clear()
    _is_valid_token.cache_clear()
