from unittest.mock import patch

from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._providers.neuronpool import NeuronpoolConversationalTask


class TestNeuronpoolProvider:
    def test_properties(self):
        helper = NeuronpoolConversationalTask()
        assert helper.provider == "neuronpool"
        assert helper.base_url == "https://api.neuronpool.dev"
        assert helper.task == "conversational"

    def test_prepare_route(self):
        helper = NeuronpoolConversationalTask()
        assert helper._prepare_route("gpt-oss-20b", "sk-neuronpool-test") == "/v1/chat/completions"

    def test_prepare_url_provider_key(self):
        helper = NeuronpoolConversationalTask()
        assert (
            helper._prepare_url("sk-neuronpool-test", "gpt-oss-20b")
            == "https://api.neuronpool.dev/v1/chat/completions"
        )

    def test_prepare_url_hf_token(self):
        helper = NeuronpoolConversationalTask()
        assert (
            helper._prepare_url("hf_test_token", "gpt-oss-20b")
            == "https://router.huggingface.co/neuronpool/v1/chat/completions"
        )

    def test_prepare_request(self):
        helper = NeuronpoolConversationalTask()
        mapping = InferenceProviderMapping(
            provider="neuronpool",
            hf_model_id="openai/gpt-oss-20b",
            providerId="gpt-oss-20b",
            task="conversational",
            status="live",
        )
        with patch.object(helper, "_prepare_mapping_info", return_value=mapping):
            request = helper.prepare_request(
                inputs=[{"role": "user", "content": "Say hello."}],
                parameters={"model": "openai/gpt-oss-20b", "temperature": 0, "max_tokens": 16},
                headers={},
                model="openai/gpt-oss-20b",
                api_key="sk-neuronpool-test",
            )

        assert request.url == "https://api.neuronpool.dev/v1/chat/completions"
        assert request.task == "conversational"
        assert request.model == "gpt-oss-20b"
        assert request.data is None
        assert request.json == {
            "model": "gpt-oss-20b",
            "messages": [{"role": "user", "content": "Say hello."}],
            "temperature": 0,
            "max_tokens": 16,
        }
        assert request.headers["authorization"] == "Bearer sk-neuronpool-test"
