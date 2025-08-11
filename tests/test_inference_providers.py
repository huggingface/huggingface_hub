import base64
import logging
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
from pytest import LogCaptureFixture

from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import RequestParameters
from huggingface_hub.inference._providers import PROVIDERS, get_provider_helper
from huggingface_hub.inference._providers._common import (
    BaseConversationalTask,
    BaseTextGenerationTask,
    TaskProviderHelper,
    filter_none,
    recursive_merge,
)
from huggingface_hub.inference._providers.black_forest_labs import BlackForestLabsTextToImageTask
from huggingface_hub.inference._providers.cohere import CohereConversationalTask
from huggingface_hub.inference._providers.fal_ai import (
    _POLLING_INTERVAL,
    FalAIAutomaticSpeechRecognitionTask,
    FalAIImageToImageTask,
    FalAIImageToVideoTask,
    FalAITextToImageTask,
    FalAITextToSpeechTask,
    FalAITextToVideoTask,
)
from huggingface_hub.inference._providers.featherless_ai import (
    FeatherlessConversationalTask,
    FeatherlessTextGenerationTask,
)
from huggingface_hub.inference._providers.fireworks_ai import FireworksAIConversationalTask
from huggingface_hub.inference._providers.groq import GroqConversationalTask
from huggingface_hub.inference._providers.hf_inference import (
    HFInferenceBinaryInputTask,
    HFInferenceConversational,
    HFInferenceFeatureExtractionTask,
    HFInferenceTask,
)
from huggingface_hub.inference._providers.hyperbolic import HyperbolicTextGenerationTask, HyperbolicTextToImageTask
from huggingface_hub.inference._providers.nebius import NebiusFeatureExtractionTask, NebiusTextToImageTask
from huggingface_hub.inference._providers.novita import NovitaConversationalTask, NovitaTextGenerationTask
from huggingface_hub.inference._providers.nscale import NscaleConversationalTask, NscaleTextToImageTask
from huggingface_hub.inference._providers.openai import OpenAIConversationalTask
from huggingface_hub.inference._providers.replicate import (
    ReplicateImageToImageTask,
    ReplicateTask,
    ReplicateTextToSpeechTask,
)
from huggingface_hub.inference._providers.sambanova import SambanovaConversationalTask, SambanovaFeatureExtractionTask
from huggingface_hub.inference._providers.together import TogetherTextToImageTask

from .testing_utils import assert_in_logs


class TestBasicTaskProviderHelper:
    def test_api_key_from_provider(self):
        helper = TaskProviderHelper(provider="provider-name", base_url="https://api.provider.com", task="task-name")
        assert helper._prepare_api_key("sk_provider_key") == "sk_provider_key"

    def test_api_key_routed(self, mocker):
        mocker.patch("huggingface_hub.inference._providers._common.get_token", return_value="hf_test_token")
        helper = TaskProviderHelper(provider="provider-name", base_url="https://api.provider.com", task="task-name")
        assert helper._prepare_api_key(None) == "hf_test_token"

    def test_api_key_missing(self):
        with patch("huggingface_hub.inference._providers._common.get_token", return_value=None):
            helper = TaskProviderHelper(
                provider="provider-name", base_url="https://api.provider.com", task="task-name"
            )
            with pytest.raises(ValueError, match="You must provide an api_key.*"):
                helper._prepare_api_key(None)

    def test_prepare_mapping_info(self, mocker, caplog: LogCaptureFixture):
        helper = TaskProviderHelper(provider="provider-name", base_url="https://api.provider.com", task="task-name")
        caplog.set_level(logging.INFO)

        # Test missing model
        with pytest.raises(ValueError, match="Please provide an HF model ID.*"):
            helper._prepare_mapping_info(None)

        # Test unsupported model
        mocker.patch(
            "huggingface_hub.inference._providers._common._fetch_inference_provider_mapping",
            return_value=[
                mocker.Mock(provider="other-provider", task="task-name", provider_id="mapped-id", status="active")
            ],
        )
        with pytest.raises(ValueError, match="Model test-model is not supported.*"):
            helper._prepare_mapping_info("test-model")

        # Test task mismatch
        mocker.patch(
            "huggingface_hub.inference._providers._common._fetch_inference_provider_mapping",
            return_value=[
                mocker.Mock(
                    task="other-task",
                    provider="provider-name",
                    providerId="mapped-id",
                    hf_model_id="test-model",
                    status="live",
                )
            ],
        )
        with pytest.raises(ValueError, match="Model test-model is not supported for task.*"):
            helper._prepare_mapping_info("test-model")

        # Test staging model
        mocker.patch(
            "huggingface_hub.inference._providers._common._fetch_inference_provider_mapping",
            return_value=[
                mocker.Mock(
                    provider="provider-name",
                    task="task-name",
                    hf_model_id="test-model",
                    provider_id="mapped-id",
                    status="staging",
                )
            ],
        )
        assert helper._prepare_mapping_info("test-model").provider_id == "mapped-id"

        assert_in_logs(
            caplog, "Model test-model is in staging mode for provider provider-name. Meant for test purposes only."
        )

        # Test successful mapping
        caplog.clear()
        mocker.patch(
            "huggingface_hub.inference._providers._common._fetch_inference_provider_mapping",
            return_value=[
                mocker.Mock(
                    provider="provider-name",
                    task="task-name",
                    hf_model_id="test-model",
                    provider_id="mapped-id",
                    status="live",
                )
            ],
        )
        assert helper._prepare_mapping_info("test-model").provider_id == "mapped-id"
        assert helper._prepare_mapping_info("test-model").hf_model_id == "test-model"
        assert helper._prepare_mapping_info("test-model").task == "task-name"
        assert helper._prepare_mapping_info("test-model").status == "live"
        assert len(caplog.records) == 0

        # Test with loras
        mocker.patch(
            "huggingface_hub.inference._providers._common._fetch_inference_provider_mapping",
            return_value=[
                mocker.Mock(
                    provider="provider-name",
                    task="task-name",
                    hf_model_id="test-model",
                    provider_id="mapped-id",
                    status="live",
                    adapter_weights_path="lora-weights-path",
                    adapter="lora",
                )
            ],
        )

        assert helper._prepare_mapping_info("test-model").adapter_weights_path == "lora-weights-path"
        assert helper._prepare_mapping_info("test-model").provider_id == "mapped-id"
        assert helper._prepare_mapping_info("test-model").hf_model_id == "test-model"
        assert helper._prepare_mapping_info("test-model").task == "task-name"
        assert helper._prepare_mapping_info("test-model").status == "live"

    def test_prepare_headers(self):
        helper = TaskProviderHelper(provider="provider-name", base_url="https://api.provider.com", task="task-name")
        headers = helper._prepare_headers({"custom": "header"}, "api_key")
        assert "user-agent" in headers  # From build_hf_headers
        assert headers["custom"] == "header"
        assert headers["authorization"] == "Bearer api_key"

    def test_prepare_url(self, mocker):
        helper = TaskProviderHelper(provider="provider-name", base_url="https://api.provider.com", task="task-name")
        mocker.patch.object(helper, "_prepare_route", return_value="/v1/test-route")

        # Test HF token routing
        url = helper._prepare_url("hf_test_token", "test-model")
        assert url == "https://router.huggingface.co/provider-name/v1/test-route"
        helper._prepare_route.assert_called_once_with("test-model", "hf_test_token")

        # Test direct API call
        helper._prepare_route.reset_mock()
        url = helper._prepare_url("sk_test_token", "test-model")
        assert url == "https://api.provider.com/v1/test-route"
        helper._prepare_route.assert_called_once_with("test-model", "sk_test_token")


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
        assert helper._prepare_route("username/repo_name", "hf_token") == "/v1/username/repo_name"

    def test_prepare_url(self):
        helper = BlackForestLabsTextToImageTask()
        assert (
            helper._prepare_url("hf_test_token", "username/repo_name")
            == "https://router.huggingface.co/black-forest-labs/v1/username/repo_name"
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
            [{"role": "user", "content": "Hello!"}],
            {},
            InferenceProviderMapping(
                provider="cohere",
                hf_model_id="CohereForAI/command-r7b-12-2024",
                providerId="CohereForAI/command-r7b-12-2024",
                task="conversational",
                status="live",
            ),
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

    def test_prepare_url(self):
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
            "a beautiful cat",
            {"width": 512, "height": 512},
            InferenceProviderMapping(
                provider="fal-ai",
                hf_model_id="username/repo_name",
                providerId="username/repo_name",
                task="text-to-image",
                status="live",
            ),
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
        assert payload == {"text": "Hello world"}

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
        mock_session = mocker.patch("huggingface_hub.inference._providers.fal_ai.get_session")
        mock_sleep = mocker.patch("huggingface_hub.inference._providers.fal_ai.time.sleep")
        mock_session.return_value.get.side_effect = [
            # First call: status
            mocker.Mock(json=lambda: {"status": "COMPLETED"}, headers={"Content-Type": "application/json"}),
            # Second call: get result
            mocker.Mock(json=lambda: {"video": {"url": "video_url"}}, headers={"Content-Type": "application/json"}),
            # Third call: get video content
            mocker.Mock(content=b"video_content"),
        ]
        api_key = helper._prepare_api_key("hf_token")
        headers = helper._prepare_headers({}, api_key)
        url = helper._prepare_url(api_key, "username/repo_name")

        request_params = RequestParameters(
            url=url,
            headers=headers,
            task="text-to-video",
            model="username/repo_name",
            data=None,
            json=None,
        )
        response = helper.get_response(
            b'{"request_id": "test_request_id", "status": "PROCESSING", "response_url": "https://queue.fal.run/username_provider/repo_name_provider/requests/test_request_id", "status_url": "https://queue.fal.run/username_provider/repo_name_provider/requests/test_request_id/status"}',
            request_params,
        )

        # Verify the correct URLs were called
        assert mock_session.return_value.get.call_count == 3
        mock_session.return_value.get.assert_has_calls(
            [
                mocker.call(
                    "https://router.huggingface.co/fal-ai/username_provider/repo_name_provider/requests/test_request_id/status?_subdomain=queue",
                    headers=request_params.headers,
                ),
                mocker.call(
                    "https://router.huggingface.co/fal-ai/username_provider/repo_name_provider/requests/test_request_id?_subdomain=queue",
                    headers=request_params.headers,
                ),
                mocker.call("video_url"),
            ]
        )
        mock_sleep.assert_called_once_with(_POLLING_INTERVAL)
        assert response == b"video_content"

    def test_image_to_image_payload(self):
        helper = FalAIImageToImageTask()
        mapping_info = InferenceProviderMapping(
            provider="fal-ai",
            hf_model_id="stabilityai/sdxl-refiner-1.0",
            providerId="fal-ai/sdxl-refiner",
            task="image-to-image",
            status="live",
        )
        payload = helper._prepare_payload_as_dict("https://example.com/image.png", {"prompt": "a cat"}, mapping_info)
        assert payload == {"image_url": "https://example.com/image.png", "prompt": "a cat"}

        payload = helper._prepare_payload_as_dict(
            b"dummy_image_data", {"prompt": "replace the cat with a dog"}, mapping_info
        )
        assert payload == {
            "image_url": f"data:image/jpeg;base64,{base64.b64encode(b'dummy_image_data').decode()}",
            "prompt": "replace the cat with a dog",
        }

    def test_image_to_image_response(self, mocker):
        helper = FalAIImageToImageTask()
        mock_session = mocker.patch("huggingface_hub.inference._providers.fal_ai.get_session")
        mock_sleep = mocker.patch("huggingface_hub.inference._providers.fal_ai.time.sleep")
        mock_session.return_value.get.side_effect = [
            # First call: status
            mocker.Mock(json=lambda: {"status": "COMPLETED"}, headers={"Content-Type": "application/json"}),
            # Second call: get result
            mocker.Mock(json=lambda: {"images": [{"url": "image_url"}]}, headers={"Content-Type": "application/json"}),
            # Third call: get image content
            mocker.Mock(content=b"image_content"),
        ]
        api_key = helper._prepare_api_key("hf_token")
        headers = helper._prepare_headers({}, api_key)
        url = helper._prepare_url(api_key, "username/repo_name")

        request_params = RequestParameters(
            url=url,
            headers=headers,
            task="image-to-image",
            model="username/repo_name",
            data=None,
            json=None,
        )
        response = helper.get_response(
            b'{"request_id": "test_request_id", "status": "PROCESSING", "response_url": "https://queue.fal.run/username_provider/repo_name_provider/requests/test_request_id", "status_url": "https://queue.fal.run/username_provider/repo_name_provider/requests/test_request_id/status"}',
            request_params,
        )

        # Verify the correct URLs were called
        assert mock_session.return_value.get.call_count == 3
        mock_session.return_value.get.assert_has_calls(
            [
                mocker.call(
                    "https://router.huggingface.co/fal-ai/username_provider/repo_name_provider/requests/test_request_id/status?_subdomain=queue",
                    headers=request_params.headers,
                ),
                mocker.call(
                    "https://router.huggingface.co/fal-ai/username_provider/repo_name_provider/requests/test_request_id?_subdomain=queue",
                    headers=request_params.headers,
                ),
                mocker.call("image_url"),
            ]
        )
        mock_sleep.assert_called_once_with(_POLLING_INTERVAL)
        assert response == b"image_content"

    def test_image_to_video_payload(self):
        helper = FalAIImageToVideoTask()
        mapping_info = InferenceProviderMapping(
            provider="fal-ai",
            hf_model_id="Wan-AI/Wan2.2-I2V-A14B",
            providerId="Wan-AI/Wan2.2-I2V-A14B",
            task="image-to-video",
            status="live",
        )
        payload = helper._prepare_payload_as_dict(
            "https://example.com/image.png",
            {"prompt": "a cat"},
            mapping_info,
        )
        assert payload == {"image_url": "https://example.com/image.png", "prompt": "a cat"}

        payload = helper._prepare_payload_as_dict(
            b"dummy_image_data",
            {"prompt": "a dog"},
            mapping_info,
        )
        assert payload == {
            "image_url": f"data:image/jpeg;base64,{base64.b64encode(b'dummy_image_data').decode()}",
            "prompt": "a dog",
        }

    def test_image_to_video_response(self, mocker):
        helper = FalAIImageToVideoTask()
        mock_session = mocker.patch("huggingface_hub.inference._providers.fal_ai.get_session")
        mock_sleep = mocker.patch("huggingface_hub.inference._providers.fal_ai.time.sleep")
        mock_session.return_value.get.side_effect = [
            # First call: status
            mocker.Mock(json=lambda: {"status": "COMPLETED"}, headers={"Content-Type": "application/json"}),
            # Second call: get result
            mocker.Mock(json=lambda: {"video": {"url": "video_url"}}, headers={"Content-Type": "application/json"}),
            # Third call: get video content
            mocker.Mock(content=b"video_content"),
        ]
        api_key = helper._prepare_api_key("hf_token")
        headers = helper._prepare_headers({}, api_key)
        url = helper._prepare_url(api_key, "username/repo_name")

        request_params = RequestParameters(
            url=url,
            headers=headers,
            task="image-to-video",
            model="username/repo_name",
            data=None,
            json=None,
        )
        response = helper.get_response(
            b'{"request_id": "test_request_id", "status": "PROCESSING", "response_url": "https://queue.fal.run/username_provider/repo_name_provider/requests/test_request_id", "status_url": "https://queue.fal.run/username_provider/repo_name_provider/requests/test_request_id/status"}',
            request_params,
        )

        # Verify the correct URLs were called
        assert mock_session.return_value.get.call_count == 3
        mock_session.return_value.get.assert_has_calls(
            [
                mocker.call(
                    "https://router.huggingface.co/fal-ai/username_provider/repo_name_provider/requests/test_request_id/status?_subdomain=queue",
                    headers=request_params.headers,
                ),
                mocker.call(
                    "https://router.huggingface.co/fal-ai/username_provider/repo_name_provider/requests/test_request_id?_subdomain=queue",
                    headers=request_params.headers,
                ),
                mocker.call("video_url"),
            ]
        )
        mock_sleep.assert_called_once_with(_POLLING_INTERVAL)
        assert response == b"video_content"


class TestFeatherlessAIProvider:
    def test_prepare_route_chat_completionurl(self):
        helper = FeatherlessConversationalTask()
        assert helper._prepare_url("rc_xxxx", "ownner/model_id") == "https://api.featherless.ai/v1/chat/completions"

        helper = FeatherlessTextGenerationTask()
        assert helper._prepare_url("rc_xxxx", "ownner/model_id") == "https://api.featherless.ai/v1/completions"


class TestFireworksAIConversationalTask:
    def test_prepare_url(self):
        helper = FireworksAIConversationalTask()
        url = helper._prepare_url("fireworks_token", "username/repo_name")
        assert url == "https://api.fireworks.ai/inference/v1/chat/completions"

    def test_prepare_payload_as_dict(self):
        helper = FireworksAIConversationalTask()
        payload = helper._prepare_payload_as_dict(
            [{"role": "user", "content": "Hello!"}],
            {},
            InferenceProviderMapping(
                provider="fireworks-ai",
                hf_model_id="meta-llama/Llama-3.1-8B-Instruct",
                providerId="meta-llama/Llama-3.1-8B-Instruct",
                task="conversational",
                status="live",
            ),
        )
        assert payload == {
            "messages": [{"role": "user", "content": "Hello!"}],
            "model": "meta-llama/Llama-3.1-8B-Instruct",
        }


class TestGroqProvider:
    def test_prepare_route(self):
        """Test route preparation for Groq conversational task."""
        helper = GroqConversationalTask()
        assert helper._prepare_route("username/repo_name", "hf_token") == "/openai/v1/chat/completions"


class TestHFInferenceProvider:
    def test_prepare_mapping_info(self, mocker):
        helper = HFInferenceTask("text-classification")
        mocker.patch(
            "huggingface_hub.inference._providers.hf_inference._check_supported_task",
            return_value=None,
        )
        mocker.patch(
            "huggingface_hub.inference._providers.hf_inference._fetch_recommended_models",
            return_value={"text-classification": "username/repo_name"},
        )
        assert helper._prepare_mapping_info("username/repo_name").provider_id == "username/repo_name"
        assert helper._prepare_mapping_info(None).provider_id == "username/repo_name"
        assert helper._prepare_mapping_info("https://any-url.com").provider_id == "https://any-url.com"

    def test_prepare_mapping_info_unknown_task(self):
        with pytest.raises(ValueError, match="Task unknown-task has no recommended model for HF Inference."):
            HFInferenceTask("unknown-task")._prepare_mapping_info(None)

    def test_prepare_url(self):
        helper = HFInferenceTask("text-classification")
        assert (
            helper._prepare_url("hf_test_token", "username/repo_name")
            == "https://router.huggingface.co/hf-inference/models/username/repo_name"
        )

        assert helper._prepare_url("hf_test_token", "https://any-url.com") == "https://any-url.com"

    def test_prepare_url_feature_extraction(self):
        helper = HFInferenceTask("feature-extraction")
        assert (
            helper._prepare_url("hf_test_token", "username/repo_name")
            == "https://router.huggingface.co/hf-inference/models/username/repo_name/pipeline/feature-extraction"
        )

    def test_prepare_url_sentence_similarity(self):
        helper = HFInferenceTask("sentence-similarity")
        assert (
            helper._prepare_url("hf_test_token", "username/repo_name")
            == "https://router.huggingface.co/hf-inference/models/username/repo_name/pipeline/sentence-similarity"
        )

    def test_prepare_payload_as_dict(self):
        helper = HFInferenceTask("text-classification")
        mapping_info = InferenceProviderMapping(
            provider="hf-inference",
            hf_model_id="username/repo_name",
            providerId="username/repo_name",
            task="text-classification",
            status="live",
        )
        assert helper._prepare_payload_as_dict(
            "dummy text input",
            parameters={"a": 1, "b": None},
            provider_mapping_info=mapping_info,
        ) == {
            "inputs": "dummy text input",
            "parameters": {"a": 1},
        }

        with pytest.raises(ValueError, match="Unexpected binary input for task text-classification."):
            helper._prepare_payload_as_dict(
                b"dummy binary data",
                {},
                mapping_info,
            )

    def test_prepare_payload_as_bytes(self):
        helper = HFInferenceBinaryInputTask("image-classification")
        mapping_info = InferenceProviderMapping(
            provider="hf-inference",
            hf_model_id="username/repo_name",
            providerId="username/repo_name",
            task="image-classification",
            status="live",
        )
        assert (
            helper._prepare_payload_as_bytes(
                b"dummy binary input",
                parameters={},
                provider_mapping_info=mapping_info,
                extra_payload=None,
            )
            == b"dummy binary input"
        )

        assert (
            helper._prepare_payload_as_bytes(
                b"dummy binary input",
                parameters={"a": 1, "b": None},
                provider_mapping_info=mapping_info,
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

    def test_prepare_request(self, mocker):
        mocker.patch(
            "huggingface_hub.inference._providers.hf_inference._check_supported_task",
            return_value=None,
        )
        mocker.patch(
            "huggingface_hub.inference._providers.hf_inference._fetch_recommended_models",
            return_value={"text-classification": "username/repo_name"},
        )
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

    def test_prepare_request_conversational(self, mocker):
        mocker.patch(
            "huggingface_hub.inference._providers.hf_inference._check_supported_task",
            return_value=None,
        )
        mocker.patch(
            "huggingface_hub.inference._providers.hf_inference._fetch_recommended_models",
            return_value={"text-classification": "username/repo_name"},
        )
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
        assert request.task == "conversational"
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
        provider_mapping_info = InferenceProviderMapping(
            provider="hf-inference",
            hf_model_id=mapped_model,
            providerId=mapped_model,
            task="conversational",
            status="live",
        )
        payload = helper._prepare_payload_as_dict(
            inputs=messages,
            parameters=parameters,
            provider_mapping_info=provider_mapping_info,
        )

        assert payload["model"] == expected_model
        assert payload["messages"] == messages

    def test_prepare_payload_feature_extraction(self):
        helper = HFInferenceFeatureExtractionTask()
        payload = helper._prepare_payload_as_dict(
            inputs="This is a test sentence.",
            parameters={"truncate": True},
            provider_mapping_info=MagicMock(),
        )
        assert payload == {"inputs": "This is a test sentence.", "truncate": True}  # not under "parameters"

    @pytest.mark.parametrize(
        "pipeline_tag,tags,task,should_raise",
        [
            # text-generation + no conversational tag -> only text-generation allowed
            (
                "text-generation",
                [],
                "text-generation",
                False,
            ),
            (
                "text-generation",
                [],
                "conversational",
                True,
            ),
            # text-generation + conversational tag -> both tasks allowed
            (
                "text-generation",
                ["conversational"],
                "text-generation",
                False,
            ),
            (
                "text-generation",
                ["conversational"],
                "conversational",
                False,
            ),
            # image-text-to-text + conversational tag -> only conversational allowed
            (
                "image-text-to-text",
                ["conversational"],
                "conversational",
                False,
            ),
            (
                "image-text-to-text",
                ["conversational"],
                "image-text-to-text",
                True,
            ),
            (
                "image-text-to-text",
                [],
                "conversational",
                True,
            ),
            # text2text-generation only allowed for text-generation task
            (
                "text2text-generation",
                [],
                "text-generation",
                False,
            ),
            (
                "text2text-generation",
                [],
                "conversational",
                True,
            ),
            # Feature-extraction / sentence-similarity are interchangeable for HF Inference
            (
                "sentence-similarity",
                ["tag1", "feature-extraction", "sentence-similarity"],
                "feature-extraction",
                False,
            ),
            (
                "feature-extraction",
                ["tag1", "feature-extraction", "sentence-similarity"],
                "sentence-similarity",
                False,
            ),
            # if pipeline_tag is not feature-extraction or sentence-similarity, raise
            ("text-generation", ["tag1", "feature-extraction", "sentence-similarity"], "sentence-similarity", True),
            # Other tasks
            (
                "audio-classification",
                [],
                "audio-classification",
                False,
            ),
            (
                "audio-classification",
                [],
                "text-classification",
                True,
            ),
        ],
    )
    def test_check_supported_task_scenarios(self, mocker, pipeline_tag, tags, task, should_raise):
        from huggingface_hub.inference._providers.hf_inference import _check_supported_task

        mock_model_info = mocker.Mock(pipeline_tag=pipeline_tag, tags=tags)
        mocker.patch("huggingface_hub.hf_api.HfApi.model_info", return_value=mock_model_info)

        if should_raise:
            with pytest.raises(ValueError):
                _check_supported_task("test-model", task)
        else:
            _check_supported_task("test-model", task)


class TestHyperbolicProvider:
    def test_prepare_route(self):
        """Test route preparation for different tasks."""
        helper = HyperbolicTextToImageTask()
        assert helper._prepare_route("username/repo_name", "hf_token") == "/v1/images/generations"

        helper = HyperbolicTextGenerationTask("text-generation")
        assert helper._prepare_route("username/repo_name", "hf_token") == "/v1/chat/completions"

        helper = HyperbolicTextGenerationTask("conversational")
        assert helper._prepare_route("username/repo_name", "hf_token") == "/v1/chat/completions"

    def test_prepare_payload_conversational(self):
        """Test payload preparation for conversational task."""
        helper = HyperbolicTextGenerationTask("conversational")
        payload = helper._prepare_payload_as_dict(
            [{"role": "user", "content": "Hello!"}],
            {"temperature": 0.7},
            InferenceProviderMapping(
                provider="hyperbolic",
                hf_model_id="meta-llama/Llama-3.2-3B-Instruct",
                providerId="meta-llama/Llama-3.2-3B-Instruct",
                task="conversational",
                status="live",
            ),
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
            InferenceProviderMapping(
                provider="hyperbolic",
                hf_model_id="stabilityai/sdxl-turbo",
                providerId="stabilityai/sdxl",
                task="text-to-image",
                status="live",
            ),
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
        assert helper._prepare_route("username/repo_name", "hf_token") == "/v1/images/generations"

    def test_prepare_payload_as_dict_text_to_image(self):
        helper = NebiusTextToImageTask()
        payload = helper._prepare_payload_as_dict(
            "a beautiful cat",
            {"num_inference_steps": 10, "width": 512, "height": 512, "guidance_scale": 7.5},
            InferenceProviderMapping(
                provider="black-forest-labs/flux-schnell",
                hf_model_id="black-forest-labs/flux-schnell",
                providerId="black-forest-labs/flux-schnell",
                task="text-to-image",
                status="live",
            ),
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

    def test_prepare_payload_as_dict_feature_extraction(self):
        helper = NebiusFeatureExtractionTask()
        payload = helper._prepare_payload_as_dict(
            "Hello world",
            {"param-that-will-be-ignored": True},
            InferenceProviderMapping(
                provider="nebius",
                hf_model_id="username/repo_name",
                providerId="provider-id",
                task="feature-extraction",
                status="live",
            ),
        )
        assert payload == {"input": "Hello world", "model": "provider-id"}

    def test_prepare_url_feature_extraction(self):
        helper = NebiusFeatureExtractionTask()
        assert (
            helper._prepare_url("hf_token", "username/repo_name")
            == "https://router.huggingface.co/nebius/v1/embeddings"
        )


class TestNovitaProvider:
    def test_prepare_url_text_generation(self):
        helper = NovitaTextGenerationTask()
        url = helper._prepare_url("novita_token", "username/repo_name")
        assert url == "https://api.novita.ai/v3/openai/completions"

    def test_prepare_url_conversational(self):
        helper = NovitaConversationalTask()
        url = helper._prepare_url("novita_token", "username/repo_name")
        assert url == "https://api.novita.ai/v3/openai/chat/completions"


class TestNscaleProvider:
    def test_prepare_route_text_to_image(self):
        helper = NscaleTextToImageTask()
        assert helper._prepare_route("model_name", "api_key") == "/v1/images/generations"

    def test_prepare_route_chat_completion(self):
        helper = NscaleConversationalTask()
        assert helper._prepare_route("model_name", "api_key") == "/v1/chat/completions"

    def test_prepare_payload_with_size_conversion(self):
        helper = NscaleTextToImageTask()
        payload = helper._prepare_payload_as_dict(
            "a beautiful landscape",
            {
                "width": 512,
                "height": 512,
            },
            InferenceProviderMapping(
                provider="nscale",
                hf_model_id="stabilityai/stable-diffusion-xl-base-1.0",
                providerId="stabilityai/stable-diffusion-xl-base-1.0",
                task="text-to-image",
                status="live",
            ),
        )
        assert payload == {
            "prompt": "a beautiful landscape",
            "size": "512x512",
            "response_format": "b64_json",
            "model": "stabilityai/stable-diffusion-xl-base-1.0",
        }

    def test_prepare_payload_as_dict(self):
        helper = NscaleTextToImageTask()
        payload = helper._prepare_payload_as_dict(
            "a beautiful landscape",
            {
                "width": 1024,
                "height": 768,
                "cfg_scale": 7.5,
                "num_inference_steps": 50,
            },
            InferenceProviderMapping(
                provider="nscale",
                hf_model_id="stabilityai/stable-diffusion-xl-base-1.0",
                providerId="stabilityai/stable-diffusion-xl-base-1.0",
                task="text-to-image",
                status="live",
            ),
        )
        assert "width" not in payload
        assert "height" not in payload
        assert "num_inference_steps" not in payload
        assert "cfg_scale" not in payload
        assert payload["size"] == "1024x768"
        assert payload["model"] == "stabilityai/stable-diffusion-xl-base-1.0"

    def test_text_to_image_get_response(self):
        helper = NscaleTextToImageTask()
        response = helper.get_response({"data": [{"b64_json": base64.b64encode(b"image_bytes").decode()}]})
        assert response == b"image_bytes"


class TestOpenAIProvider:
    def test_prepare_url(self):
        helper = OpenAIConversationalTask()
        assert helper._prepare_url("sk-XXXXXX", "gpt-4o-mini") == "https://api.openai.com/v1/chat/completions"


class TestReplicateProvider:
    def test_prepare_headers(self):
        helper = ReplicateTask("text-to-image")
        headers = helper._prepare_headers({}, "my_replicate_key")
        headers["Prefer"] == "wait"
        headers["authorization"] == "Bearer my_replicate_key"

    def test_prepare_route(self):
        helper = ReplicateTask("text-to-image")

        # No model version
        url = helper._prepare_route("black-forest-labs/FLUX.1-schnell", "hf_token")
        assert url == "/v1/models/black-forest-labs/FLUX.1-schnell/predictions"

        # Model with specific version
        url = helper._prepare_route("black-forest-labs/FLUX.1-schnell:1944af04d098ef", "hf_token")
        assert url == "/v1/predictions"

    def test_prepare_payload_as_dict(self):
        helper = ReplicateTask("text-to-image")

        # No model version
        payload = helper._prepare_payload_as_dict(
            "a beautiful cat",
            {"num_inference_steps": 20},
            InferenceProviderMapping(
                provider="replicate",
                hf_model_id="black-forest-labs/FLUX.1-schnell",
                providerId="black-forest-labs/FLUX.1-schnell",
                task="text-to-image",
                status="live",
            ),
        )
        assert payload == {"input": {"prompt": "a beautiful cat", "num_inference_steps": 20}}

        # Model with specific version
        payload = helper._prepare_payload_as_dict(
            "a beautiful cat",
            {"num_inference_steps": 20},
            InferenceProviderMapping(
                provider="replicate",
                hf_model_id="black-forest-labs/FLUX.1-schnell",
                providerId="black-forest-labs/FLUX.1-schnell:1944af04d098ef",
                task="text-to-image",
                status="live",
            ),
        )
        assert payload == {
            "input": {"prompt": "a beautiful cat", "num_inference_steps": 20},
            "version": "1944af04d098ef",
        }

    def test_text_to_speech_payload(self):
        helper = ReplicateTextToSpeechTask()
        payload = helper._prepare_payload_as_dict(
            "Hello world",
            {},
            InferenceProviderMapping(
                provider="replicate",
                hf_model_id="hexgrad/Kokoro-82M",
                providerId="hexgrad/Kokoro-82M:f559560eb822dc509045f3921a1921234918b91739db4bf3daab2169b71c7a13",
                task="text-to-speech",
                status="live",
            ),
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

    def test_image_to_image_payload(self):
        helper = ReplicateImageToImageTask()
        dummy_image = b"dummy image data"
        encoded_image = base64.b64encode(dummy_image).decode("utf-8")
        image_uri = f"data:image/jpeg;base64,{encoded_image}"

        # No model version
        payload = helper._prepare_payload_as_dict(
            dummy_image,
            {"num_inference_steps": 20},
            InferenceProviderMapping(
                provider="replicate",
                hf_model_id="google/gemini-pro-vision",
                providerId="google/gemini-pro-vision",
                task="image-to-image",
                status="live",
            ),
        )
        assert payload == {
            "input": {"input_image": image_uri, "num_inference_steps": 20},
        }

        payload = helper._prepare_payload_as_dict(
            dummy_image,
            {"num_inference_steps": 20},
            InferenceProviderMapping(
                provider="replicate",
                hf_model_id="google/gemini-pro-vision",
                providerId="google/gemini-pro-vision:123456",
                task="image-to-image",
                status="live",
            ),
        )
        assert payload == {
            "input": {"input_image": image_uri, "num_inference_steps": 20},
            "version": "123456",
        }


class TestSambanovaProvider:
    def test_prepare_url_conversational(self):
        helper = SambanovaConversationalTask()
        assert (
            helper._prepare_url("sambanova_token", "username/repo_name")
            == "https://api.sambanova.ai/v1/chat/completions"
        )

    def test_prepare_payload_as_dict_feature_extraction(self):
        helper = SambanovaFeatureExtractionTask()
        payload = helper._prepare_payload_as_dict(
            "Hello world",
            {"truncate": True},
            InferenceProviderMapping(
                provider="sambanova",
                hf_model_id="username/repo_name",
                providerId="provider-id",
                task="feature-extraction",
                status="live",
            ),
        )
        assert payload == {"input": "Hello world", "model": "provider-id", "truncate": True}

    def test_prepare_url_feature_extraction(self):
        helper = SambanovaFeatureExtractionTask()
        assert (
            helper._prepare_url("hf_token", "username/repo_name")
            == "https://router.huggingface.co/sambanova/v1/embeddings"
        )


class TestTogetherProvider:
    def test_prepare_route_text_to_image(self):
        helper = TogetherTextToImageTask()
        assert helper._prepare_route("username/repo_name", "hf_token") == "/v1/images/generations"

    def test_prepare_payload_as_dict_text_to_image(self):
        helper = TogetherTextToImageTask()
        payload = helper._prepare_payload_as_dict(
            "a beautiful cat",
            {"num_inference_steps": 10, "guidance_scale": 1, "width": 512, "height": 512},
            InferenceProviderMapping(
                provider="together",
                hf_model_id="black-forest-labs/FLUX.1-schnell",
                providerId="black-forest-labs/FLUX.1-schnell",
                task="text-to-image",
                status="live",
            ),
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
        assert helper._prepare_route("dummy-model", "hf_token") == "/v1/chat/completions"
        assert helper.task == "conversational"

    def test_prepare_payload(self):
        helper = BaseConversationalTask(provider="test-provider", base_url="https://api.test.com")
        messages = [{"role": "user", "content": "Hello!"}]
        parameters = {"temperature": 0.7, "max_tokens": 100}

        payload = helper._prepare_payload_as_dict(
            inputs=messages,
            parameters=parameters,
            provider_mapping_info=InferenceProviderMapping(
                provider="test-provider",
                hf_model_id="test-model",
                providerId="test-provider-id",
                task="conversational",
                status="live",
            ),
        )

        assert payload == {
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 100,
            "model": "test-provider-id",
        }

    @pytest.mark.parametrize(
        "raw_messages, expected_messages",
        [
            (
                [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": None,
                    }
                ],
                [
                    {
                        "role": "assistant",
                        "content": "",
                    }
                ],
            ),
            (
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_current_weather",
                                    "arguments": '{"location": "San Francisco, CA", "unit": "celsius"}',
                                },
                            },
                        ],
                    },
                    {
                        "role": "tool",
                        "content": "pong",
                        "tool_call_id": "abc123",
                        "name": "dummy_tool",
                        "tool_calls": None,
                    },
                ],
                [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_current_weather",
                                    "arguments": '{"location": "San Francisco, CA", "unit": "celsius"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": "pong",
                        "tool_call_id": "abc123",
                        "name": "dummy_tool",
                    },
                ],
            ),
        ],
    )
    def test_prepare_payload_filters_messages(self, raw_messages, expected_messages):
        helper = BaseConversationalTask(provider="test-provider", base_url="https://api.test.com")

        parameters = {
            "temperature": 0.2,
            "max_tokens": None,
            "top_p": None,
        }

        payload = helper._prepare_payload_as_dict(
            inputs=raw_messages,
            parameters=parameters,
            provider_mapping_info=InferenceProviderMapping(
                provider="test-provider",
                hf_model_id="test-model",
                providerId="test-provider-id",
                task="conversational",
                status="live",
            ),
        )

        assert payload["messages"] == expected_messages
        assert payload["temperature"] == 0.2
        assert "max_tokens" not in payload
        assert "top_p" not in payload


class TestBaseTextGenerationTask:
    def test_prepare_route(self):
        helper = BaseTextGenerationTask(provider="test-provider", base_url="https://api.test.com")
        assert helper._prepare_route("dummy-model", "hf_token") == "/v1/completions"
        assert helper.task == "text-generation"

    def test_prepare_payload(self):
        helper = BaseTextGenerationTask(provider="test-provider", base_url="https://api.test.com")
        prompt = "Once upon a time"
        parameters = {"temperature": 0.7, "max_tokens": 100}

        payload = helper._prepare_payload_as_dict(
            inputs=prompt,
            parameters=parameters,
            provider_mapping_info=InferenceProviderMapping(
                provider="test-provider",
                hf_model_id="test-model",
                providerId="test-provider-id",
                task="text-generation",
                status="live",
            ),
        )

        assert payload == {
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 100,
            "model": "test-provider-id",
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


@pytest.mark.parametrize(
    "data, expected",
    [
        ({}, {}),  # empty dictionary remains empty
        ({"a": 1, "b": None, "c": 3}, {"a": 1, "c": 3}),  # remove None at root level
        ({"a": None, "b": {"x": None, "y": 2}}, {"b": {"y": 2}}),  # remove nested None
        ({"a": {"b": {"c": None}}}, {"a": {"b": {}}}),  # keep empty nested dict
        (
            {"a": "", "b": {"x": {"y": None}, "z": 0}, "c": []},  # do not remove 0, [] and "" values
            {"a": "", "b": {"x": {}, "z": 0}, "c": []},
        ),
        (
            {"a": [0, 1, None]},  # do not remove None in lists
            {"a": [0, 1, None]},
        ),
        # dicts inside list are cleaned, list level None kept
        ({"a": [{"x": None, "y": 1}, None]}, {"a": [{"y": 1}, None]}),
        # remove every None that is the value of a dict key
        (
            [None, {"x": None, "y": 5}, [None, 6]],
            [None, {"y": 5}, [None, 6]],
        ),
        ({"a": [None, {"x": None}]}, {"a": [None, {}]}),
    ],
)
def test_filter_none(data: Dict, expected: Dict):
    """Test that filter_none removes None values from nested dictionaries."""
    assert filter_none(data) == expected


def test_get_provider_helper_auto(mocker):
    """Test the 'auto' provider selection logic."""

    mock_provider_a_helper = mocker.Mock(spec=TaskProviderHelper)
    mock_provider_b_helper = mocker.Mock(spec=TaskProviderHelper)
    PROVIDERS["provider-a"] = {"test-task": mock_provider_a_helper}
    PROVIDERS["provider-b"] = {"test-task": mock_provider_b_helper}

    mocker.patch(
        "huggingface_hub.inference._providers._fetch_inference_provider_mapping",
        return_value=[
            mocker.Mock(provider="provider-a"),
            mocker.Mock(provider="provider-b"),
        ],
    )
    helper = get_provider_helper(provider="auto", task="test-task", model="test-model")

    # The helper should be the one from provider-a
    assert helper is mock_provider_a_helper

    PROVIDERS.pop("provider-a", None)
    PROVIDERS.pop("provider-b", None)
