import base64
from typing import Dict

import pytest

from huggingface_hub.inference._providers._common import (
    BaseConversationalTask,
    BaseTextGenerationTask,
    recursive_merge,
)
from huggingface_hub.inference._providers.black_forest_labs import BlackForestLabsTextToImageTask
from huggingface_hub.inference._providers.cohere import CohereConversationalTask
from huggingface_hub.inference._providers.fal_ai import (
    FalAIAutomaticSpeechRecognitionTask,
    FalAITextToImageTask,
    FalAITextToSpeechTask,
    FalAITextToVideoTask,
)
from huggingface_hub.inference._providers.fireworks_ai import FireworksAIConversationalTask
from huggingface_hub.inference._providers.hf_inference import (
    HFInferenceBinaryInputTask,
    HFInferenceConversational,
    HFInferenceTask,
)
from huggingface_hub.inference._providers.hyperbolic import (
    HyperbolicTextGenerationTask,
    HyperbolicTextToImageTask,
)
from huggingface_hub.inference._providers.nebius import NebiusTextToImageTask
from huggingface_hub.inference._providers.novita import (
    NovitaConversationalTask,
    NovitaTextGenerationTask,
)
from huggingface_hub.inference._providers.replicate import ReplicateTask, ReplicateTextToSpeechTask
from huggingface_hub.inference._providers.sambanova import SambanovaConversationalTask
from huggingface_hub.inference._providers.together import (
    TogetherTextToImageTask,
)


class TestBlackForestLabsProvider:
    def test_prepare_headers_bfl_key(self):
        helper = BlackForestLabsTextToImageTask()
        headers = helper._prepare_headers({}, "bfl_key")
        assert "authorization" not in headers
        assert headers["X-Key"] == "bfl_key"

    def test_prepare_headers_hf_key(self):
        """When using HF token, must use Bearer authorization."""
        helper = BlackForestLabsTextToImageTask()
        headers = helper._prepare_headers({}, "hf_test_token")
        assert headers["authorization"] == "Bearer hf_test_token"
        assert "X-Key" not in headers

    def test_prepare_route(self):
        """Test route preparation."""
        helper = BlackForestLabsTextToImageTask()
        assert helper._prepare_route("username/repo_name") == "username/repo_name"

    def test_prepare_url(self):
        helper = BlackForestLabsTextToImageTask()
        assert (
            helper._prepare_url("hf_test_token", "username/repo_name")
            == "https://router.huggingface.co/black-forest-labs/username/repo_name"
        )

    def test_prepare_payload_as_dict(self):
        """Test payload preparation with parameter renaming."""
        helper = BlackForestLabsTextToImageTask()
        payload = helper._prepare_payload_as_dict(
            "a beautiful cat",
            {
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "width": 512,
                "height": 512,
                "seed": 42,
            },
            "username/repo_name",
        )
        assert payload == {
            "prompt": "a beautiful cat",
            "steps": 30,  # renamed from num_inference_steps
            "guidance": 7.5,  # renamed from guidance_scale
            "width": 512,
            "height": 512,
            "seed": 42,
        }

    def test_get_response_success(self, mocker):
        """Test successful response handling with polling."""
        helper = BlackForestLabsTextToImageTask()
        mock_session = mocker.patch("huggingface_hub.inference._providers.black_forest_labs.get_session")
        mock_session.return_value.get.side_effect = [
            mocker.Mock(
                json=lambda: {"status": "Ready", "result": {"sample": "https://example.com/image.jpg"}},
                raise_for_status=lambda: None,
            ),
            mocker.Mock(content=b"image_bytes", raise_for_status=lambda: None),
        ]

        response = helper.get_response({"polling_url": "https://example.com/poll"})

        assert response == b"image_bytes"
        assert mock_session.return_value.get.call_count == 2
        mock_session.return_value.get.assert_has_calls(
            [
                mocker.call("https://example.com/poll", headers={"Content-Type": "application/json"}),
                mocker.call("https://example.com/image.jpg"),
            ]
        )


class TestCohereConversationalTask:
    def test_prepare_url(self):
        helper = CohereConversationalTask()
        assert helper.task == "conversational"
        url = helper._prepare_url("cohere_token", "username/repo_name")
        assert url == "https://api.cohere.com/compatibility/v1/chat/completions"

    def test_prepare_payload_as_dict(self):
        helper = CohereConversationalTask()
        payload = helper._prepare_payload_as_dict(
            [{"role": "user", "content": "Hello!"}], {}, "CohereForAI/command-r7b-12-2024"
        )
        assert payload == {
            "messages": [{"role": "user", "content": "Hello!"}],
            "model": "CohereForAI/command-r7b-12-2024",
        }


class TestFalAIProvider:
    def test_prepare_headers_fal_ai_key(self):
        """When using direct call, must use Key authorization."""
        headers = FalAITextToImageTask()._prepare_headers({}, "fal_ai_key")
        assert headers["authorization"] == "Key fal_ai_key"

    def test_prepare_headers_hf_key(self):
        """When using routed call, must use Bearer authorization."""
        headers = FalAITextToImageTask()._prepare_headers({}, "hf_token")
        assert headers["authorization"] == "Bearer hf_token"

    def test_prepare_route(self):
        url = FalAITextToImageTask()._prepare_url("hf_token", "username/repo_name")
        assert url == "https://router.huggingface.co/fal-ai/username/repo_name"

    def test_automatic_speech_recognition_payload(self):
        helper = FalAIAutomaticSpeechRecognitionTask()
        payload = helper._prepare_payload_as_dict("https://example.com/audio.mp3", {}, "username/repo_name")
        assert payload == {"audio_url": "https://example.com/audio.mp3"}

        payload = helper._prepare_payload_as_dict(b"dummy_audio_data", {}, "username/repo_name")
        assert payload == {"audio_url": f"data:audio/mpeg;base64,{base64.b64encode(b'dummy_audio_data').decode()}"}

    def test_automatic_speech_recognition_response(self):
        helper = FalAIAutomaticSpeechRecognitionTask()
        response = helper.get_response({"text": "Hello world"})
        assert response == "Hello world"

        with pytest.raises(ValueError):
            helper.get_response({"text": 123})

    def test_text_to_image_payload(self):
        helper = FalAITextToImageTask()
        payload = helper._prepare_payload_as_dict(
            "a beautiful cat", {"width": 512, "height": 512}, "username/repo_name"
        )
        assert payload == {
            "prompt": "a beautiful cat",
            "image_size": {"width": 512, "height": 512},
        }

    def test_text_to_image_response(self, mocker):
        helper = FalAITextToImageTask()
        mock = mocker.patch("huggingface_hub.inference._providers.fal_ai.get_session")
        response = helper.get_response({"images": [{"url": "image_url"}]})
        mock.return_value.get.assert_called_once_with("image_url")
        assert response == mock.return_value.get.return_value.content

    def test_text_to_speech_payload(self):
        helper = FalAITextToSpeechTask()
        payload = helper._prepare_payload_as_dict("Hello world", {}, "username/repo_name")
        assert payload == {"lyrics": "Hello world"}

    def test_text_to_speech_response(self, mocker):
        helper = FalAITextToSpeechTask()
        mock = mocker.patch("huggingface_hub.inference._providers.fal_ai.get_session")
        response = helper.get_response({"audio": {"url": "audio_url"}})
        mock.return_value.get.assert_called_once_with("audio_url")
        assert response == mock.return_value.get.return_value.content

    def test_text_to_video_payload(self):
        helper = FalAITextToVideoTask()
        payload = helper._prepare_payload_as_dict("a cat walking", {"num_frames": 16}, "username/repo_name")
        assert payload == {"prompt": "a cat walking", "num_frames": 16}

    def test_text_to_video_response(self, mocker):
        helper = FalAITextToVideoTask()
        mock = mocker.patch("huggingface_hub.inference._providers.fal_ai.get_session")
        response = helper.get_response({"video": {"url": "video_url"}})
        mock.return_value.get.assert_called_once_with("video_url")
        assert response == mock.return_value.get.return_value.content


class TestFireworksAIConversationalTask:
    def test_prepare_url(self):
        helper = FireworksAIConversationalTask()
        url = helper._prepare_url("fireworks_token", "username/repo_name")
        assert url == "https://api.fireworks.ai/inference/v1/chat/completions"

    def test_prepare_payload_as_dict(self):
        helper = FireworksAIConversationalTask()
        payload = helper._prepare_payload_as_dict(
            [{"role": "user", "content": "Hello!"}], {}, "meta-llama/Llama-3.1-8B-Instruct"
        )
        assert payload == {
            "messages": [{"role": "user", "content": "Hello!"}],
            "model": "meta-llama/Llama-3.1-8B-Instruct",
        }


class TestHFInferenceProvider:
    def test_prepare_mapped_model(self, mocker):
        helper = HFInferenceTask("text-classification")
        assert helper._prepare_mapped_model("username/repo_name") == "username/repo_name"
        assert helper._prepare_mapped_model("https://any-url.com") == "https://any-url.com"

        mocker.patch(
            "huggingface_hub.inference._providers.hf_inference._fetch_recommended_models",
            return_value={"text-classification": "username/repo_name"},
        )
        assert helper._prepare_mapped_model(None) == "username/repo_name"

        with pytest.raises(ValueError, match="Task unknown-task has no recommended model"):
            assert HFInferenceTask("unknown-task")._prepare_mapped_model(None)

    def test_prepare_url(self):
        helper = HFInferenceTask("text-classification")
        assert (
            helper._prepare_url("hf_test_token", "username/repo_name")
            == "https://router.huggingface.co/hf-inference/models/username/repo_name"
        )

        assert helper._prepare_url("hf_test_token", "https://any-url.com") == "https://any-url.com"

    def test_prepare_payload_as_dict(self):
        helper = HFInferenceTask("text-classification")
        assert helper._prepare_payload_as_dict(
            "dummy text input",
            parameters={"a": 1, "b": None},
            mapped_model="username/repo_name",
        ) == {
            "inputs": "dummy text input",
            "parameters": {"a": 1},
        }

        with pytest.raises(ValueError, match="Unexpected binary input for task text-classification."):
            helper._prepare_payload_as_dict(b"dummy binary data", {}, "username/repo_name")

    def test_prepare_payload_as_bytes(self):
        helper = HFInferenceBinaryInputTask("image-classification")
        assert (
            helper._prepare_payload_as_bytes(
                b"dummy binary input",
                parameters={},
                mapped_model="username/repo_name",
                extra_payload=None,
            )
            == b"dummy binary input"
        )

        assert (
            helper._prepare_payload_as_bytes(
                b"dummy binary input",
                parameters={"a": 1, "b": None},
                mapped_model="username/repo_name",
                extra_payload={"extra": "payload"},
            )
            == b'{"inputs": "ZHVtbXkgYmluYXJ5IGlucHV0", "parameters": {"a": 1}, "extra": "payload"}'
            # base64.b64encode(b"dummy binary input")
        )

    def test_conversational_url(self):
        helper = HFInferenceConversational()
        helper._prepare_url(
            "hf_test_token", "username/repo_name"
        ) == "https://router.huggingface.co/hf-inference/models/username/repo_name/v1/chat/completions"
        helper._prepare_url("hf_test_token", "https://any-url.com") == "https://any-url.com/v1/chat/completions"
        helper._prepare_url("hf_test_token", "https://any-url.com/v1") == "https://any-url.com/v1/chat/completions"

    def test_prepare_request(self):
        helper = HFInferenceTask("text-classification")
        request = helper.prepare_request(
            inputs="this is a dummy input",
            parameters={},
            headers={},
            model="username/repo_name",
            api_key="hf_test_token",
        )

        assert request.url == "https://router.huggingface.co/hf-inference/models/username/repo_name"
        assert request.task == "text-classification"
        assert request.model == "username/repo_name"
        assert request.headers["authorization"] == "Bearer hf_test_token"
        assert request.json == {"inputs": "this is a dummy input", "parameters": {}}

    def test_prepare_request_conversational(self):
        helper = HFInferenceConversational()
        request = helper.prepare_request(
            inputs=[{"role": "user", "content": "dummy text input"}],
            parameters={},
            headers={},
            model="username/repo_name",
            api_key="hf_test_token",
        )

        assert (
            request.url == "https://router.huggingface.co/hf-inference/models/username/repo_name/v1/chat/completions"
        )
        assert request.task == "text-generation"
        assert request.model == "username/repo_name"
        assert request.json == {
            "model": "username/repo_name",
            "messages": [{"role": "user", "content": "dummy text input"}],
        }

    @pytest.mark.parametrize(
        "mapped_model,parameters,expected_model",
        [
            (
                "username/repo_name",
                {},
                "username/repo_name",
            ),
            # URL endpoint with model in parameters - use model from parameters
            (
                "http://localhost:8000/v1/chat/completions",
                {"model": "username/repo_name"},
                "username/repo_name",
            ),
            # URL endpoint without model - fallback to dummy
            (
                "http://localhost:8000/v1/chat/completions",
                {},
                "dummy",
            ),
            # HTTPS endpoint with model in parameters
            (
                "https://api.example.com/v1/chat/completions",
                {"model": "username/repo_name"},
                "username/repo_name",
            ),
            # URL endpoint with other parameters - should still use dummy
            (
                "http://localhost:8000/v1/chat/completions",
                {"temperature": 0.7, "max_tokens": 100},
                "dummy",
            ),
        ],
    )
    def test_prepare_payload_as_dict_conversational(self, mapped_model, parameters, expected_model):
        helper = HFInferenceConversational()
        messages = [{"role": "user", "content": "Hello!"}]

        payload = helper._prepare_payload_as_dict(
            inputs=messages,
            parameters=parameters,
            mapped_model=mapped_model,
        )

        assert payload["model"] == expected_model
        assert payload["messages"] == messages


class TestHyperbolicProvider:
    def test_prepare_route(self):
        """Test route preparation for different tasks."""
        helper = HyperbolicTextToImageTask()
        assert helper._prepare_route("username/repo_name") == "/v1/images/generations"

        helper = HyperbolicTextGenerationTask("text-generation")
        assert helper._prepare_route("username/repo_name") == "/v1/chat/completions"

        helper = HyperbolicTextGenerationTask("conversational")
        assert helper._prepare_route("username/repo_name") == "/v1/chat/completions"

    def test_prepare_payload_conversational(self):
        """Test payload preparation for conversational task."""
        helper = HyperbolicTextGenerationTask("conversational")
        payload = helper._prepare_payload_as_dict(
            [{"role": "user", "content": "Hello!"}], {"temperature": 0.7}, "meta-llama/Llama-3.2-3B-Instruct"
        )
        assert payload == {
            "messages": [{"role": "user", "content": "Hello!"}],
            "temperature": 0.7,
            "model": "meta-llama/Llama-3.2-3B-Instruct",
        }

    def test_prepare_payload_text_to_image(self):
        """Test payload preparation for text-to-image task."""
        helper = HyperbolicTextToImageTask()
        payload = helper._prepare_payload_as_dict(
            "a beautiful cat",
            {
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "width": 512,
                "height": 512,
                "seed": 42,
            },
            "stabilityai/sdxl",
        )
        assert payload == {
            "prompt": "a beautiful cat",
            "steps": 30,  # renamed from num_inference_steps
            "cfg_scale": 7.5,  # renamed from guidance_scale
            "width": 512,
            "height": 512,
            "seed": 42,
            "model_name": "stabilityai/sdxl",
        }

    def test_text_to_image_get_response(self):
        """Test response handling for text-to-image task."""
        helper = HyperbolicTextToImageTask()
        dummy_image = b"image_bytes"
        response = helper.get_response({"images": [{"image": base64.b64encode(dummy_image).decode()}]})
        assert response == dummy_image


class TestNebiusProvider:
    def test_prepare_route_text_to_image(self):
        helper = NebiusTextToImageTask()
        assert helper._prepare_route("username/repo_name") == "/v1/images/generations"

    def test_prepare_payload_as_dict_text_to_image(self):
        helper = NebiusTextToImageTask()
        payload = helper._prepare_payload_as_dict(
            "a beautiful cat",
            {"num_inference_steps": 10, "width": 512, "height": 512, "guidance_scale": 7.5},
            "black-forest-labs/flux-schnell",
        )
        assert payload == {
            "prompt": "a beautiful cat",
            "response_format": "b64_json",
            "width": 512,
            "height": 512,
            "num_inference_steps": 10,
            "model": "black-forest-labs/flux-schnell",
        }

    def test_text_to_image_get_response(self):
        helper = NebiusTextToImageTask()
        response = helper.get_response({"data": [{"b64_json": base64.b64encode(b"image_bytes").decode()}]})
        assert response == b"image_bytes"


class TestNovitaProvider:
    def test_prepare_url_text_generation(self):
        helper = NovitaTextGenerationTask()
        url = helper._prepare_url("novita_token", "username/repo_name")
        assert url == "https://api.novita.ai/v3/openai/completions"

    def test_prepare_url_conversational(self):
        helper = NovitaConversationalTask()
        url = helper._prepare_url("novita_token", "username/repo_name")
        assert url == "https://api.novita.ai/v3/openai/chat/completions"


class TestReplicateProvider:
    def test_prepare_headers(self):
        helper = ReplicateTask("text-to-image")
        headers = helper._prepare_headers({}, "my_replicate_key")
        headers["Prefer"] == "wait"
        headers["authorization"] == "Bearer my_replicate_key"

    def test_prepare_route(self):
        helper = ReplicateTask("text-to-image")

        # No model version
        url = helper._prepare_route("black-forest-labs/FLUX.1-schnell")
        assert url == "/v1/models/black-forest-labs/FLUX.1-schnell/predictions"

        # Model with specific version
        url = helper._prepare_route("black-forest-labs/FLUX.1-schnell:1944af04d098ef")
        assert url == "/v1/predictions"

    def test_prepare_payload_as_dict(self):
        helper = ReplicateTask("text-to-image")

        # No model version
        payload = helper._prepare_payload_as_dict(
            "a beautiful cat", {"num_inference_steps": 20}, "black-forest-labs/FLUX.1-schnell"
        )
        assert payload == {"input": {"prompt": "a beautiful cat", "num_inference_steps": 20}}

        # Model with specific version
        payload = helper._prepare_payload_as_dict(
            "a beautiful cat", {"num_inference_steps": 20}, "black-forest-labs/FLUX.1-schnell:1944af04d098ef"
        )
        assert payload == {
            "input": {"prompt": "a beautiful cat", "num_inference_steps": 20},
            "version": "1944af04d098ef",
        }

    def test_text_to_speech_payload(self):
        helper = ReplicateTextToSpeechTask()
        payload = helper._prepare_payload_as_dict(
            "Hello world", {}, "hexgrad/Kokoro-82M:f559560eb822dc509045f3921a1921234918b91739db4bf3daab2169b71c7a13"
        )
        assert payload == {
            "input": {"text": "Hello world"},
            "version": "f559560eb822dc509045f3921a1921234918b91739db4bf3daab2169b71c7a13",
        }

    def test_get_response_timeout(self):
        helper = ReplicateTask("text-to-image")
        with pytest.raises(TimeoutError, match="Inference request timed out after 60 seconds."):
            helper.get_response({"model": "black-forest-labs/FLUX.1-schnell"})  # no 'output' key

    def test_get_response_single_output(self, mocker):
        helper = ReplicateTask("text-to-image")
        mock = mocker.patch("huggingface_hub.inference._providers.replicate.get_session")
        response = helper.get_response({"output": "https://example.com/image.jpg"})
        mock.return_value.get.assert_called_once_with("https://example.com/image.jpg")
        assert response == mock.return_value.get.return_value.content


class TestSambanovaProvider:
    def test_prepare_url(self):
        helper = SambanovaConversationalTask()
        assert (
            helper._prepare_url("sambanova_token", "username/repo_name")
            == "https://api.sambanova.ai/v1/chat/completions"
        )


class TestTogetherProvider:
    def test_prepare_route_text_to_image(self):
        helper = TogetherTextToImageTask()
        assert helper._prepare_route("username/repo_name") == "/v1/images/generations"

    def test_prepare_payload_as_dict_text_to_image(self):
        helper = TogetherTextToImageTask()
        payload = helper._prepare_payload_as_dict(
            "a beautiful cat",
            {"num_inference_steps": 10, "guidance_scale": 1, "width": 512, "height": 512},
            "black-forest-labs/FLUX.1-schnell",
        )
        assert payload == {
            "prompt": "a beautiful cat",
            "response_format": "base64",
            "width": 512,
            "height": 512,
            "steps": 10,  # renamed field
            "guidance": 1,  # renamed field
            "model": "black-forest-labs/FLUX.1-schnell",
        }

    def test_text_to_image_get_response(self):
        helper = TogetherTextToImageTask()
        response = helper.get_response({"data": [{"b64_json": base64.b64encode(b"image_bytes").decode()}]})
        assert response == b"image_bytes"


class TestBaseConversationalTask:
    def test_prepare_route(self):
        helper = BaseConversationalTask(provider="test-provider", base_url="https://api.test.com")
        assert helper._prepare_route("dummy-model") == "/v1/chat/completions"
        assert helper.task == "conversational"

    def test_prepare_payload(self):
        helper = BaseConversationalTask(provider="test-provider", base_url="https://api.test.com")
        messages = [{"role": "user", "content": "Hello!"}]
        parameters = {"temperature": 0.7, "max_tokens": 100}

        payload = helper._prepare_payload_as_dict(
            inputs=messages,
            parameters=parameters,
            mapped_model="test-model",
        )

        assert payload == {
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 100,
            "model": "test-model",
        }


class TestBaseTextGenerationTask:
    def test_prepare_route(self):
        helper = BaseTextGenerationTask(provider="test-provider", base_url="https://api.test.com")
        assert helper._prepare_route("dummy-model") == "/v1/completions"
        assert helper.task == "text-generation"

    def test_prepare_payload(self):
        helper = BaseTextGenerationTask(provider="test-provider", base_url="https://api.test.com")
        prompt = "Once upon a time"
        parameters = {"temperature": 0.7, "max_tokens": 100}

        payload = helper._prepare_payload_as_dict(
            inputs=prompt,
            parameters=parameters,
            mapped_model="test-model",
        )

        assert payload == {
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 100,
            "model": "test-model",
        }


@pytest.mark.parametrize(
    "dict1, dict2, expected",
    [
        # Basic merge with non-overlapping keys
        ({"a": 1}, {"b": 2}, {"a": 1, "b": 2}),
        # Overwriting a key
        ({"a": 1}, {"a": 2}, {"a": 2}),
        # Empty dict merge
        ({}, {"a": 1}, {"a": 1}),
        ({"a": 1}, {}, {"a": 1}),
        ({}, {}, {}),
        # Nested dictionary merge
        (
            {"a": {"b": 1}},
            {"a": {"c": 2}},
            {"a": {"b": 1, "c": 2}},
        ),
        # Overwriting nested dictionary key
        (
            {"a": {"b": 1}},
            {"a": {"b": 2}},
            {"a": {"b": 2}},
        ),
        # Deep merge
        (
            {"a": {"b": {"c": 1}}},
            {"a": {"b": {"d": 2}}},
            {"a": {"b": {"c": 1, "d": 2}}},
        ),
        # Overwriting a nested value with a non-dict type
        (
            {"a": {"b": {"c": 1}}},
            {"a": {"b": 2}},
            {"a": {"b": 2}},  # Overwrites dict with integer
        ),
        # Merging dictionaries with different types
        (
            {"a": 1},
            {"a": {"b": 2}},
            {"a": {"b": 2}},  # Overwrites int with dict
        ),
    ],
)
def test_recursive_merge(dict1: Dict, dict2: Dict, expected: Dict):
    initial_dict1 = dict1.copy()
    initial_dict2 = dict2.copy()
    assert recursive_merge(dict1, dict2) == expected
    # does not mutate the inputs
    assert dict1 == initial_dict1
    assert dict2 == initial_dict2
