import base64

import pytest

from huggingface_hub.inference._providers.fal_ai import (
    FalAIAutomaticSpeechRecognitionTask,
    FalAITextToImageTask,
    FalAITextToVideoTask,
)
from huggingface_hub.inference._providers.hf_inference import (
    HFInferenceBinaryInputTask,
    HFInferenceConversational,
    HFInferenceTask,
)
from huggingface_hub.inference._providers.replicate import ReplicateTask, ReplicateTextToSpeechTask
from huggingface_hub.inference._providers.sambanova import SambanovaConversationalTask
from huggingface_hub.inference._providers.together import TogetherTextGenerationTask, TogetherTextToImageTask


@pytest.fixture(autouse=True)
def patch_inference_proxy_template(monkeypatch):
    monkeypatch.setattr(
        "huggingface_hub.constants.INFERENCE_PROXY_TEMPLATE",
        "https://huggingface.co/api/inference-proxy/{provider}",
    )


class TestHFInferenceProvider:
    def test_prepare_request(self):
        helper = HFInferenceTask("text-classification")
        request = helper.prepare_request(
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

    # Testing conversational task separately
    def test_prepare_request_conversational(self):
        helper = HFInferenceConversational()
        request = helper.prepare_request(
            inputs=[{"role": "user", "content": "dummy text input"}],
            parameters={},
            headers={},
            model="username/repo_name",
            api_key="hf_test_token",
        )

        assert request.url == "https://api-inference.huggingface.co/models/username/repo_name/v1/chat/completions"
        assert request.task == "text-generation"
        assert request.model == "username/repo_name"
        assert request.json == {
            "model": "username/repo_name",
            "messages": [{"role": "user", "content": "dummy text input"}],
        }

    @pytest.mark.parametrize(
        "helper,inputs,parameters,expected_data,expected_json",
        [
            (
                HFInferenceTask("text-classification"),
                "dummy text input",
                {},
                None,
                {
                    "inputs": "dummy text input",
                    "parameters": {},
                },
            ),
            (
                HFInferenceBinaryInputTask("image-classification"),
                b"dummy binary data",
                {},
                b"dummy binary data",
                None,
            ),
            (
                HFInferenceBinaryInputTask("text-to-image"),
                b"dummy binary data",
                {"threshold": 0.9},
                None,
                {
                    "inputs": f"{base64.b64encode(b'dummy binary data').decode()}",
                    "parameters": {"threshold": 0.9},
                },
            ),
        ],
        ids=["text-task", "binary-raw", "binary-with-params"],
    )
    def test_prepare_payload(self, helper, inputs, parameters, expected_data, expected_json):
        data, json = helper._prepare_payload(inputs, parameters, model=None, extra_payload={})
        assert data == expected_data
        assert json == expected_json

    def test_get_response(self):
        pytest.skip("Not implemented yet")


class TestFalAIProvider:
    def test_prepare_request(self):
        helper = FalAITextToImageTask()

        # Test with custom fal.ai key
        request = helper.prepare_request(
            inputs="dummy text input",
            parameters={},
            headers={},
            model="black-forest-labs/FLUX.1-dev",
            api_key="my_fal_ai_key",
        )
        assert request.url.startswith("https://fal.run/")
        assert request.headers["authorization"] == "Key my_fal_ai_key"

        # Test with missing token
        with pytest.raises(ValueError, match="You must provide an api_key to work with fal.ai API."):
            helper.prepare_request(
                inputs="dummy text input",
                parameters={},
                headers={},
                model="black-forest-labs/FLUX.1-dev",
                api_key=None,
            )

        # Test routing
        request = helper.prepare_request(
            inputs="dummy text input",
            parameters={},
            headers={},
            model="black-forest-labs/FLUX.1-dev",
            api_key="hf_test_token",
        )
        assert request.headers["authorization"] == "Bearer hf_test_token"
        assert request.url.startswith("https://huggingface.co/api/inference-proxy/fal-ai")

    @pytest.mark.parametrize(
        "helper,inputs,parameters,expected_payload",
        [
            (
                FalAITextToImageTask(),
                "a beautiful cat",
                {"width": 512, "height": 512},
                {
                    "prompt": "a beautiful cat",
                    "image_size": {"width": 512, "height": 512},
                },
            ),
            (
                FalAITextToVideoTask(),
                "a cat walking",
                {"num_frames": 16},
                {
                    "prompt": "a cat walking",
                    "num_frames": 16,
                },
            ),
            (
                FalAIAutomaticSpeechRecognitionTask(),
                "https://example.com/audio.mp3",
                {},
                {
                    "audio_url": "https://example.com/audio.mp3",
                },
            ),
            (
                FalAIAutomaticSpeechRecognitionTask(),
                b"dummy audio data",
                {},
                {
                    "audio_url": f"data:audio/mpeg;base64,{base64.b64encode(b'dummy audio data').decode()}",
                },
            ),
        ],
        ids=["text-to-image", "text-to-video", "speech-recognition-url", "speech-recognition-binary"],
    )
    def test_prepare_payload(self, helper, inputs, parameters, expected_payload):
        payload = helper._prepare_payload(inputs, parameters)
        assert payload == expected_payload

    def test_get_response(self):
        pytest.skip("Not implemented yet")


class TestReplicateProvider:
    def test_prepare_request(self):
        helper = ReplicateTask("text-to-image")

        # Test with custom replicate key
        request = helper.prepare_request(
            inputs="dummy text input",
            parameters={},
            headers={},
            model="black-forest-labs/FLUX.1-schnell",
            api_key="my_replicate_key",
        )
        assert request.url.startswith("https://api.replicate.com/")
        assert request.headers["Prefer"] == "wait"

        # Test with missing token
        with pytest.raises(ValueError, match="You must provide an api_key to work with Replicate API."):
            helper.prepare_request(
                inputs="dummy text input",
                parameters={},
                headers={},
                model="black-forest-labs/FLUX.1-schnell",
                api_key=None,
            )

        # Test routing
        request = helper.prepare_request(
            inputs="dummy text input",
            parameters={},
            headers={},
            model="black-forest-labs/FLUX.1-schnell",
            api_key="hf_test_token",
        )
        assert request.url.startswith("https://huggingface.co/api/inference-proxy/replicate")

    @pytest.mark.parametrize(
        "helper,model,inputs,parameters,expected_payload",
        [
            (
                ReplicateTask("text-to-image"),
                "black-forest-labs/FLUX.1-schnell",
                "a beautiful cat",
                {"num_inference_steps": 20},
                {
                    "input": {
                        "prompt": "a beautiful cat",
                        "num_inference_steps": 20,
                    }
                },
            ),
            (
                ReplicateTextToSpeechTask(),
                "OuteAI/OuteTTS-0.3-500M",
                "Hello world",
                {},
                {
                    "input": {
                        "inputs": "Hello world",
                    },
                    "version": "39a59319327b27327fa3095149c5a746e7f2aee18c75055c3368237a6503cd26",
                },
            ),
            (
                ReplicateTask("text-to-video"),
                "genmo/mochi-1-preview",
                "a cat walking",
                {"num_frames": 16},
                {
                    "input": {
                        "prompt": "a cat walking",
                        "num_frames": 16,
                    },
                    "version": "1944af04d098ef69bed7f9d335d102e652203f268ec4aaa2d836f6217217e460",
                },
            ),
        ],
        ids=["text-to-image", "text-to-speech", "text-to-video"],
    )
    def test_prepare_payload(self, helper, model, inputs, parameters, expected_payload):
        mapped_model = helper._map_model(model)
        payload = helper._prepare_payload(inputs, parameters, mapped_model)
        assert payload == expected_payload

    def test_get_response(self):
        pytest.skip("Not implemented yet")


class TestTogetherProvider:
    def test_prepare_request(self):
        helper = TogetherTextGenerationTask("conversational")
        # Test with custom together key
        request = helper.prepare_request(
            inputs="this is a dummy input",
            parameters={},
            headers={},
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            api_key="my_together_key",
        )
        assert request.url.startswith("https://api.together.xyz/")
        assert request.model == "meta-llama/Llama-3-70b-chat-hf"

        # Test with missing token
        with pytest.raises(ValueError, match="You must provide an api_key to work with Together API."):
            helper.prepare_request(
                inputs="this is a dummy input",
                parameters={},
                headers={},
                model="meta-llama/Meta-Llama-3-70B-Instruct",
                api_key=None,
            )

        # Test routing
        request = helper.prepare_request(
            inputs="this is a dummy input",
            parameters={},
            headers={},
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            api_key="hf_test_token",
        )
        assert request.url.startswith("https://huggingface.co/api/inference-proxy/together")

    @pytest.mark.parametrize(
        "helper,inputs,parameters,expected_payload",
        [
            (
                TogetherTextGenerationTask("conversational"),
                [{"role": "user", "content": "Hello!"}],
                {},
                {
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            ),
            (
                TogetherTextToImageTask(),
                "a beautiful image of a cat",
                {
                    "num_inference_steps": 25,
                    "guidance_scale": 7,
                    "target_size": (512, 512),
                },
                {
                    "prompt": "a beautiful image of a cat",
                    "response_format": "base64",
                    "steps": 25,
                    "guidance": 7,
                    "width": 512,
                    "height": 512,
                },
            ),
        ],
        ids=["conversational", "text-to-image"],
    )
    def test_prepare_payload(self, helper, inputs, parameters, expected_payload):
        payload = helper._prepare_payload(inputs, parameters)
        assert payload == expected_payload

    def test_get_response(self):
        pytest.skip("Not implemented yet")


class TestSambanovaProvider:
    def test_prepare_request(self):
        helper = SambanovaConversationalTask()
        # Test with custom sambanova key
        request = helper.prepare_request(
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
            helper.prepare_request(
                inputs="this is a dummy input",
                parameters={},
                headers={},
                model="meta-llama/Llama-3.1-8B-Instruct",
                api_key=None,
            )

        # Test routing
        request = helper.prepare_request(
            inputs="this is a dummy input",
            parameters={},
            headers={},
            model="meta-llama/Llama-3.1-8B-Instruct",
            api_key="hf_test_token",
        )
        assert request.url.startswith("https://huggingface.co/api/inference-proxy/sambanova")

    def test_get_response(self):
        pytest.skip("Not implemented yet")
