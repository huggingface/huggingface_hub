import pytest

from huggingface_hub.inference._providers.fal_ai import FalAITextToImageTask
from huggingface_hub.inference._providers.hf_inference import HFInferenceTask
from huggingface_hub.inference._providers.replicate import ReplicateTextToImageTask
from huggingface_hub.inference._providers.sambanova import SambanovaConversationalTask
from huggingface_hub.inference._providers.together import TogetherTextGenerationTask


@pytest.fixture(autouse=True)
def patch_inference_proxy_template(monkeypatch):
    monkeypatch.setattr(
        "huggingface_hub.constants.INFERENCE_PROXY_TEMPLATE",
        "https://huggingface.co/api/inference-proxy/{provider}",
    )


class TestHFInferenceProvider:
    helper = HFInferenceTask("text-classification")

    def test_prepare_request(self):
        request = self.helper.prepare_request(
            inputs="this is a dummy input",
            parameters={},
            headers={},
            model="username/repo_name",
            api_key="hf_test_token",
        )

        assert request.url == "https://api-inference.huggingface.co/models/username/repo_name"
        assert request.task == "text-classification"
        assert request.model == "username/repo_name"
        assert request.headers["authorization"] == "Bearer hf_test_token"
        assert request.json == {"inputs": "this is a dummy input", "parameters": {}}

    def test_get_response(self):
        pytest.skip("Not implemented yet")


class TestFalAIProvider:
    helper = FalAITextToImageTask()

    def test_prepare_request(self):
        # Test with custom fal.ai key
        request = self.helper.prepare_request(
            inputs="this is a dummy input",
            parameters={},
            headers={},
            model="black-forest-labs/FLUX.1-dev",
            api_key="my_fal_ai_key",
        )
        assert request.url.startswith("https://fal.run/")
        assert request.model == "fal-ai/flux/dev"
        assert request.headers["authorization"] == "Key my_fal_ai_key"
        assert "prompt" in request.json

        # Test with missing token
        with pytest.raises(ValueError, match="You must provide an api_key to work with fal.ai API."):
            self.helper.prepare_request(
                inputs="this is a dummy input",
                parameters={},
                headers={},
                model="black-forest-labs/FLUX.1-dev",
                api_key=None,
            )

        # Test routing
        request = self.helper.prepare_request(
            inputs="this is a dummy input",
            parameters={},
            headers={},
            model="black-forest-labs/FLUX.1-dev",
            api_key="hf_test_token",
        )
        assert request.url.startswith("https://huggingface.co/api/inference-proxy/fal-ai")
        assert request.headers["authorization"] == "Bearer hf_test_token"

    def test_get_response(self):
        pytest.skip("Not implemented yet")


class TestReplicateProvider:
    helper = ReplicateTextToImageTask()

    def test_prepare_request(self):
        # Test with custom replicate key
        request = self.helper.prepare_request(
            inputs="this is a dummy input",
            parameters={},
            headers={},
            model="black-forest-labs/FLUX.1-schnell",
            api_key="my_replicate_key",
        )
        assert request.url.startswith("https://api.replicate.com/")
        assert request.model == "black-forest-labs/flux-schnell"
        assert "input" in request.json
        assert "prompt" in request.json["input"]

        # Test with missing token
        with pytest.raises(ValueError, match="You must provide an api_key to work with Replicate API."):
            self.helper.prepare_request(
                inputs="this is a dummy input",
                parameters={},
                headers={},
                model="black-forest-labs/FLUX.1-schnell",
                api_key=None,
            )

        # Test routing
        request = self.helper.prepare_request(
            inputs="this is a dummy input",
            parameters={},
            headers={},
            model="black-forest-labs/FLUX.1-schnell",
            api_key="hf_test_token",
        )
        assert request.url.startswith("https://huggingface.co/api/inference-proxy/replicate")

    def test_get_response(self):
        pytest.skip("Not implemented yet")


class TestTogetherProvider:
    helper = TogetherTextGenerationTask("conversational")

    def test_prepare_request(self):
        # Test with custom together key
        request = self.helper.prepare_request(
            inputs="this is a dummy input",
            parameters={},
            headers={},
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            api_key="my_together_key",
        )
        assert request.url.startswith("https://api.together.xyz/")
        assert request.model == "meta-llama/Llama-3-70b-chat-hf"
        assert "messages" in request.json

        # Test with missing token
        with pytest.raises(ValueError, match="You must provide an api_key to work with Together API."):
            self.helper.prepare_request(
                inputs="this is a dummy input",
                parameters={},
                headers={},
                model="meta-llama/Meta-Llama-3-70B-Instruct",
                api_key=None,
            )

        # Test routing
        request = self.helper.prepare_request(
            inputs="this is a dummy input",
            parameters={},
            headers={},
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            api_key="hf_test_token",
        )
        assert request.url.startswith("https://huggingface.co/api/inference-proxy/together")

    def test_get_response(self):
        pytest.skip("Not implemented yet")


class TestSambanovaProvider:
    helper = SambanovaConversationalTask()

    def test_prepare_request(self):
        # Test with custom sambanova key
        request = self.helper.prepare_request(
            inputs="this is a dummy input",
            parameters={},
            headers={},
            model="meta-llama/Llama-3.1-8B-Instruct",
            api_key="my_sambanova_key",
        )
        assert request.url.startswith("https://api.sambanova.ai/")
        assert request.model == "Meta-Llama-3.1-8B-Instruct"
        assert "messages" in request.json

        # Test with missing token
        with pytest.raises(ValueError, match="You must provide an api_key to work with Sambanova API."):
            self.helper.prepare_request(
                inputs="this is a dummy input",
                parameters={},
                headers={},
                model="meta-llama/Llama-3.1-8B-Instruct",
                api_key=None,
            )

        # Test routing
        request = self.helper.prepare_request(
            inputs="this is a dummy input",
            parameters={},
            headers={},
            model="meta-llama/Llama-3.1-8B-Instruct",
            api_key="hf_test_token",
        )
        assert request.url.startswith("https://huggingface.co/api/inference-proxy/sambanova")

    def test_get_response(self):
        pytest.skip("Not implemented yet")
