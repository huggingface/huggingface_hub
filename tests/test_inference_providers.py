import base64
from typing import Dict

import pytest

from huggingface_hub.inference._providers._common import recursive_merge
from huggingface_hub.inference._providers.fal_ai import (
    FalAIAutomaticSpeechRecognitionTask,
    FalAITextToImageTask,
    FalAITextToSpeechTask,
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
        assert url == "https://router.huggingface.co/fal-ai//username/repo_name"

    def test_automatic_speech_recognition_payload(self):
        helper = FalAIAutomaticSpeechRecognitionTask()
        payload = helper._prepare_payload("https://example.com/audio.mp3", {}, "username/repo_name")
        assert payload == {"audio_url": "https://example.com/audio.mp3"}

        payload = helper._prepare_payload(b"dummy_audio_data", {}, "username/repo_name")
        assert payload == {"audio_url": f"data:audio/mpeg;base64,{base64.b64encode(b'dummy_audio_data').decode()}"}

    def test_automatic_speech_recognition_response(self):
        helper = FalAIAutomaticSpeechRecognitionTask()
        response = helper.get_response({"text": "Hello world"})
        assert response == "Hello world"

        with pytest.raises(ValueError):
            helper.get_response({"text": 123})

    def test_text_to_image_payload(self):
        helper = FalAITextToImageTask()
        payload = helper._prepare_payload("a beautiful cat", {"width": 512, "height": 512}, "username/repo_name")
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
        payload = helper._prepare_payload("Hello world", {}, "username/repo_name")
        assert payload == {"lyrics": "Hello world"}

    def test_text_to_speech_response(self, mocker):
        helper = FalAITextToSpeechTask()
        mock = mocker.patch("huggingface_hub.inference._providers.fal_ai.get_session")
        response = helper.get_response({"audio": {"url": "audio_url"}})
        mock.return_value.get.assert_called_once_with("audio_url")
        assert response == mock.return_value.get.return_value.content

    def test_text_to_video_payload(self):
        helper = FalAITextToVideoTask()
        payload = helper._prepare_payload("a cat walking", {"num_frames": 16}, "username/repo_name")
        assert payload == {"prompt": "a cat walking", "num_frames": 16}

    def test_text_to_video_response(self, mocker):
        helper = FalAITextToVideoTask()
        mock = mocker.patch("huggingface_hub.inference._providers.fal_ai.get_session")
        response = helper.get_response({"video": {"url": "video_url"}})
        mock.return_value.get.assert_called_once_with("video_url")
        assert response == mock.return_value.get.return_value.content


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

    def test_prepare_payload(self):
        helper = HFInferenceTask("text-classification")
        assert helper._prepare_payload(
            "dummy text input",
            parameters={"a": 1, "b": None},
            mapped_model="username/repo_name",
        ) == {
            "inputs": "dummy text input",
            "parameters": {"a": 1},
        }

        with pytest.raises(ValueError, match="Unexpected binary input for task text-classification."):
            helper._prepare_payload(b"dummy binary data", {}, "username/repo_name")

    def test_prepare_body_binary_input(self):
        helper = HFInferenceBinaryInputTask("image-classification")
        assert (
            helper._prepare_body(
                b"dummy binary input",
                parameters={},
                mapped_model="username/repo_name",
                extra_payload=None,
            )
            == b"dummy binary input"
        )

        assert (
            helper._prepare_body(
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

    def test_prepare_payload(self):
        helper = ReplicateTask("text-to-image")

        # No model version
        payload = helper._prepare_payload(
            "a beautiful cat", {"num_inference_steps": 20}, "black-forest-labs/FLUX.1-schnell"
        )
        assert payload == {"input": {"prompt": "a beautiful cat", "num_inference_steps": 20}}

        # Model with specific version
        payload = helper._prepare_payload(
            "a beautiful cat", {"num_inference_steps": 20}, "black-forest-labs/FLUX.1-schnell:1944af04d098ef"
        )
        assert payload == {
            "input": {"prompt": "a beautiful cat", "num_inference_steps": 20},
            "version": "1944af04d098ef",
        }

    def test_text_to_speech_payload(self):
        helper = ReplicateTextToSpeechTask()
        payload = helper._prepare_payload(
            "Hello world", {}, "hexgrad/Kokoro-82M:f559560eb822dc509045f3921a1921234918b91739db4bf3daab2169b71c7a13"
        )
        assert payload == {
            "input": {"text": "Hello world"},
            "version": "f559560eb822dc509045f3921a1921234918b91739db4bf3daab2169b71c7a13",
        }


class TestSambanovaProvider:
    def test_prepare_route(self):
        helper = SambanovaConversationalTask()
        assert helper._prepare_route("meta-llama/Llama-3.1-8B-Instruct") == "/v1/chat/completions"

    def test_prepare_payload(self):
        helper = SambanovaConversationalTask()
        payload = helper._prepare_payload(
            [{"role": "user", "content": "Hello!"}], {}, "meta-llama/Llama-3.1-8B-Instruct"
        )
        assert payload == {
            "messages": [{"role": "user", "content": "Hello!"}],
            "model": "meta-llama/Llama-3.1-8B-Instruct",
        }


class TestTogetherProvider:
    def test_prepare_route(self):
        helper = TogetherTextGenerationTask("text-generation")
        assert helper._prepare_route("username/repo_name") == "/v1/completions"

        helper = TogetherTextGenerationTask("conversational")
        assert helper._prepare_route("username/repo_name") == "/v1/chat/completions"

        helper = TogetherTextToImageTask()
        assert helper._prepare_route("username/repo_name") == "/v1/images/generations"

    def test_prepare_payload_conversational(self):
        helper = TogetherTextGenerationTask("conversational")
        payload = helper._prepare_payload(
            [{"role": "user", "content": "Hello!"}], {}, "meta-llama/Llama-3.1-8B-Instruct"
        )
        assert payload == {
            "messages": [{"role": "user", "content": "Hello!"}],
            "model": "meta-llama/Llama-3.1-8B-Instruct",
        }

    def test_prepare_payload_text_to_image(self):
        helper = TogetherTextToImageTask()
        payload = helper._prepare_payload(
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
