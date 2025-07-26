import pytest

from huggingface_hub.inference._providers.bagelnet import BagelNetConversationalTask


class TestBagelNetConversationalTask:
    def test_init(self):
        """Test BagelNet provider initialization."""
        task = BagelNetConversationalTask()
        assert task.provider == "bagelnet"
        assert task.base_url == "https://api.bagel.net"
        assert task.task == "conversational"

    def test_inheritance(self):
        """Test BagelNet inherits from BaseConversationalTask."""
        from huggingface_hub.inference._providers._common import BaseConversationalTask
        
        task = BagelNetConversationalTask()
        assert isinstance(task, BaseConversationalTask)

    def test_no_method_overrides(self):
        """Test that BagelNet uses default implementations (no overrides needed)."""
        task = BagelNetConversationalTask()
        
        # Should use default route
        route = task._prepare_route("test_model", "test_key")
        assert route == "/v1/chat/completions"
        
        # Should use default base URL behavior  
        direct_url = task._prepare_base_url("sk-test-key")  # Non-HF key
        assert direct_url == "https://api.bagel.net" 