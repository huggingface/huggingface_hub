import pytest

from huggingface_hub.constants import HF_HUB_DOWNLOAD_MODES, _hf_hub_download_mode


@pytest.fixture
def _clear_download_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_DOWNLOAD_MODE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)


def test_allowed_modes_match_literal() -> None:
    assert HF_HUB_DOWNLOAD_MODES == ("auto", "prefer_offline", "offline")


@pytest.mark.usefixtures("_clear_download_mode_env")
def test_default_is_auto() -> None:
    assert _hf_hub_download_mode() == "auto"


@pytest.mark.usefixtures("_clear_download_mode_env")
@pytest.mark.parametrize(
    "value,expected",
    [
        ("auto", "auto"),
        ("AUTO", "auto"),
        ("prefer_offline", "prefer_offline"),
        ("PREFER_OFFLINE", "prefer_offline"),
        ("offline", "offline"),
        ("OFFLINE", "offline"),
    ],
)
def test_download_mode_wins_over_legacy_offline(monkeypatch: pytest.MonkeyPatch, value: str, expected: str) -> None:
    monkeypatch.setenv("HF_HUB_DOWNLOAD_MODE", value)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    assert _hf_hub_download_mode() == expected


@pytest.mark.usefixtures("_clear_download_mode_env")
@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", "offline"),
        ("true", "offline"),
        ("YES", "offline"),
        ("0", "auto"),
        ("false", "auto"),
    ],
)
def test_legacy_hf_hub_offline_when_mode_unset(monkeypatch: pytest.MonkeyPatch, value: str, expected: str) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", value)
    assert _hf_hub_download_mode() == expected


@pytest.mark.usefixtures("_clear_download_mode_env")
def test_legacy_hf_hub_offline_zero_wins_over_transformers_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    assert _hf_hub_download_mode() == "auto"


@pytest.mark.usefixtures("_clear_download_mode_env")
def test_transformers_offline_when_mode_and_hf_hub_offline_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    assert _hf_hub_download_mode() == "offline"


@pytest.mark.usefixtures("_clear_download_mode_env")
def test_empty_download_mode_falls_back_to_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_DOWNLOAD_MODE", "  ")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert _hf_hub_download_mode() == "offline"


@pytest.mark.usefixtures("_clear_download_mode_env")
def test_invalid_download_mode_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_DOWNLOAD_MODE", "nope")
    with pytest.raises(ValueError, match="Invalid HF_HUB_DOWNLOAD_MODE"):
        _hf_hub_download_mode()
