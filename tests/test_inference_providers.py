import base64
import logging
from unittest.mock import MagicMock, patch

import pytest
from pytest import LogCaptureFixture

from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import RequestParameters
from huggingface_hub.inference._providers import PROVIDERS, get_provider_helper
from huggingface_hub.inference._providers._common import (
    AutoRouterConversationalTask,
    BaseConversationalTask,
    BaseTextGenerationTask,
    TaskProviderHelper,
    filter_none,
    recursive_merge,
)
from huggingface_hub.inference._providers.black_forest_labs import BlackForestLabsTextToImageTask
from huggingface_hub.inference._providers.clarifai import ClarifaiConversationalTask
from huggingface_hub.inference._providers.cohere import CohereConversationalTask
from huggingface_hub.inference._providers.fal_ai import (
    _POLLING_INTERVAL,
    FalAIAutomaticSpeechRecognitionTask,
    FalAIImageSegmentationTask,
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
from huggingface_hub.inference._providers.ovhcloud import OVHcloudConversationalTask
from huggingface_hub.inference._providers.publicai import PublicAIConversationalTask
from huggingface_hub.inference._providers.replicate import (
    ReplicateAutomaticSpeechRecognitionTask,
    ReplicateImageToImageTask,
    ReplicateTask,
    ReplicateTextToSpeechTask,
)
from huggingface_hub.inference._providers.sambanova import SambanovaConversationalTask, SambanovaFeatureExtractionTask
from huggingface_hub.inference._providers.scaleway import ScalewayConversationalTask, ScalewayFeatureExtractionTask
from huggingface_hub.inference._providers.together import TogetherTextToImageTask
from huggingface_hub.inference._providers.wavespeed import (
    WavespeedAIImageToImageTask,
    WavespeedAIImageToVideoTask,
    WavespeedAITextToImageTask,
    WavespeedAITextToVideoTask,
)
from huggingface_hub.inference._providers.zai_org import _POLLING_INTERVAL as ZAI_POLLING_INTERVAL
from huggingface_hub.inference._providers.zai_org import ZaiConversationalTask, ZaiTextToImageTask

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


class TestAutoRouterConversationalTask:
    def test_properties(self):
        helper = AutoRouterConversationalTask()
        assert helper.provider == "auto"
        assert helper.base_url == "https://router.huggingface.co"
        assert helper.task == "conversational"

    def test_prepare_mapping_info_is_fake(self):
        helper = AutoRouterConversationalTask()
        mapping_info = helper._prepare_mapping_info("test-model")
        assert mapping_info.hf_model_id == "test-model"
        assert mapping_info.provider_id == "test-model"
        assert mapping_info.task == "conversational"
        assert mapping_info.status == "live"

    def test_prepare_request(self):
        helper = AutoRouterConversationalTask()

        request = helper.prepare_request(
            inputs=[{"role": "user", "content": "Hello!"}],
            parameters={"model": "test-model", "frequency_penalty": 1.0},
            headers={},
            model="test-model",
            api_key="hf_test_token",
        )

        # Use auto-router URL
        assert request.url == "https://router.huggingface.co/v1/chat/completions"

        # The rest is the expected request for a Chat Completion API
        assert request.headers["authorization"] == "Bearer hf_test_token"
        assert request.json == {
            "messages": [{"role": "user", "content": "Hello!"}],
            "model": "test-model",
            "frequency_penalty": 1.0,
        }
        assert request.task == "conversational"
        assert request.model == "test-model"
        assert request.data is None


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


class TestClarifaiProvider:
    def test_prepare_url(self):
        helper = ClarifaiConversationalTask()
        assert (
            helper._prepare_url("clarifai_api_key", "username/repo_name")
            == "https://api.clarifai.com/v2/ext/openai/v1/chat/completions"
        )

    def test_prepare_payload_as_dict(self):
        helper = ClarifaiConversationalTask()
        payload = helper._prepare_payload_as_dict(
            [{"role": "user", "content": "Hello!"}],
            {},
            InferenceProviderMapping(
                provider="clarifai",
                hf_model_id="meta-llama/llama-3.1-8B-Instruct",
                providerId="meta-llama/llama-3.1-8B-Instruct",
                task="conversational",
                status="live",
            ),
        )

        assert payload == {
            "messages": [{"role": "user", "content": "Hello!"}],
            "model": "meta-llama/llama-3.1-8B-Instruct",
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
        assert response == {"text": "Hello world"}

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

    def test_image_segmentation_payload(self):
        helper = FalAIImageSegmentationTask()
        mapping_info = InferenceProviderMapping(
            provider="fal-ai",
            hf_model_id="briaai/RMBG-2.0",
            providerId="fal-ai/rmbg-2.0",
            task="image-segmentation",
            status="live",
        )
        payload = helper._prepare_payload_as_dict("https://example.com/image.png", {"threshold": 0.5}, mapping_info)
        assert payload == {"image_url": "https://example.com/image.png", "threshold": 0.5, "sync_mode": True}

        payload = helper._prepare_payload_as_dict(b"dummy_image_data", {"mask_threshold": 0.8}, mapping_info)
        assert payload == {
            "image_url": f"data:image/png;base64,{base64.b64encode(b'dummy_image_data').decode()}",
            "mask_threshold": 0.8,
            "sync_mode": True,
        }

    def test_image_segmentation_response_with_data_url(self, mocker):
        """Test image segmentation response when image URL is a data URL."""
        helper = FalAIImageSegmentationTask()
        mock_session = mocker.patch("huggingface_hub.inference._providers.fal_ai.get_session")
        mock_sleep = mocker.patch("huggingface_hub.inference._providers.fal_ai.time.sleep")
        dummy_mask_base64 = base64.b64encode(b"mask_content").decode()
        data_url = f"data:image/png;base64,{dummy_mask_base64}"
        mock_session.return_value.get.side_effect = [
            # First call: status
            mocker.Mock(json=lambda: {"status": "COMPLETED"}, headers={"Content-Type": "application/json"}),
            # Second call: get result
            mocker.Mock(json=lambda: {"image": {"url": data_url}}, headers={"Content-Type": "application/json"}),
        ]
        api_key = helper._prepare_api_key("hf_token")
        headers = helper._prepare_headers({}, api_key)
        url = helper._prepare_url(api_key, "username/repo_name")

        request_params = RequestParameters(
            url=url,
            headers=headers,
            task="image-segmentation",
            model="username/repo_name",
            data=None,
            json=None,
        )
        response = helper.get_response(
            b'{"request_id": "test_request_id", "status": "PROCESSING", "response_url": "https://queue.fal.run/username_provider/repo_name_provider/requests/test_request_id", "status_url": "https://queue.fal.run/username_provider/repo_name_provider/requests/test_request_id/status"}',
            request_params,
        )

        # Verify the correct URLs were called (only status and result, no fetch needed for data URL)
        assert mock_session.return_value.get.call_count == 2
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
            ]
        )
        mock_sleep.assert_called_once_with(_POLLING_INTERVAL)
        assert response == [{"label": "mask", "mask": dummy_mask_base64}]

    def test_image_segmentation_response_with_regular_url(self, mocker):
        """Test image segmentation response when image URL is a regular HTTP URL."""
        helper = FalAIImageSegmentationTask()
        mock_session = mocker.patch("huggingface_hub.inference._providers.fal_ai.get_session")
        mock_sleep = mocker.patch("huggingface_hub.inference._providers.fal_ai.time.sleep")
        dummy_mask_base64 = base64.b64encode(b"mask_content").decode()
        mock_session.return_value.get.side_effect = [
            # First call: status
            mocker.Mock(json=lambda: {"status": "COMPLETED"}, headers={"Content-Type": "application/json"}),
            # Second call: get result
            mocker.Mock(
                json=lambda: {"image": {"url": "https://example.com/mask.png"}},
                headers={"Content-Type": "application/json"},
            ),
            # Third call: get mask content
            mocker.Mock(content=b"mask_content", raise_for_status=lambda: None),
        ]
        api_key = helper._prepare_api_key("hf_token")
        headers = helper._prepare_headers({}, api_key)
        url = helper._prepare_url(api_key, "username/repo_name")

        request_params = RequestParameters(
            url=url,
            headers=headers,
            task="image-segmentation",
            model="username/repo_name",
            data=None,
            json=None,
        )
        response = helper.get_response(
            b'{"request_id": "test_request_id", "status": "PROCESSING", "response_url": "https://queue.fal.run/username_provider/repo_name_provider/requests/test_request_id", "status_url": "https://queue.fal.run/username_provider/repo_name_provider/requests/test_request_id/status"}',
            request_params,
        )

        # Verify the correct URLs were called (status, result, and mask fetch)
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
                mocker.call("https://example.com/mask.png"),
            ]
        )
        mock_sleep.assert_called_once_with(_POLLING_INTERVAL)
        assert response == [{"label": "mask", "mask": dummy_mask_base64}]


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

    def test_prepare_request_from_binary_data(self, mocker, tmp_path):
        helper = HFInferenceBinaryInputTask("image-classification")

        mock_model_info = mocker.Mock(pipeline_tag="image-classification", tags=[])
        mocker.patch("huggingface_hub.hf_api.HfApi.model_info", return_value=mock_model_info)

        image_path = tmp_path / "image.jpg"
        image_path.write_bytes(b"dummy binary input")

        request = helper.prepare_request(
            inputs=image_path,
            parameters={},
            headers={},
            model="microsoft/resnet-50",
            api_key="hf_test_token",
            extra_payload=None,
        )
        assert request.url == "https://router.huggingface.co/hf-inference/models/microsoft/resnet-50"
        assert request.task == "image-classification"
        assert request.model == "microsoft/resnet-50"
        assert request.json is None
        assert isinstance(request.data, bytes)
        assert request.headers["authorization"] == "Bearer hf_test_token"
        assert request.headers["content-type"] == "image/jpeg"  # based on filename


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


class TestScalewayProvider:
    def test_prepare_hf_url_conversational(self):
        helper = ScalewayConversationalTask()
        url = helper._prepare_url("hf_token", "username/repo_name")
        assert url == "https://router.huggingface.co/scaleway/v1/chat/completions"

    def test_prepare_url_conversational(self):
        helper = ScalewayConversationalTask()
        url = helper._prepare_url("scw_token", "username/repo_name")
        assert url == "https://api.scaleway.ai/v1/chat/completions"

    def test_prepare_payload_as_dict(self):
        helper = ScalewayConversationalTask()
        payload = helper._prepare_payload_as_dict(
            [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello!"},
            ],
            {
                "max_tokens": 512,
                "temperature": 0.15,
                "top_p": 1,
                "presence_penalty": 0,
                "stream": True,
            },
            InferenceProviderMapping(
                provider="scaleway",
                hf_model_id="meta-llama/Llama-3.1-8B-Instruct",
                providerId="meta-llama/llama-3.1-8B-Instruct",
                task="conversational",
                status="live",
            ),
        )
        assert payload == {
            "max_tokens": 512,
            "messages": [
                {"content": "You are a helpful assistant", "role": "system"},
                {"role": "user", "content": "Hello!"},
            ],
            "model": "meta-llama/llama-3.1-8B-Instruct",
            "presence_penalty": 0,
            "stream": True,
            "temperature": 0.15,
            "top_p": 1,
        }

    def test_prepare_url_feature_extraction(self):
        helper = ScalewayFeatureExtractionTask()
        assert (
            helper._prepare_url("hf_token", "username/repo_name")
            == "https://router.huggingface.co/scaleway/v1/embeddings"
        )

    def test_prepare_payload_as_dict_feature_extraction(self):
        helper = ScalewayFeatureExtractionTask()
        payload = helper._prepare_payload_as_dict(
            "Example text to embed",
            {"truncate": True},
            InferenceProviderMapping(
                provider="scaleway",
                hf_model_id="username/repo_name",
                providerId="provider-id",
                task="feature-extraction",
                status="live",
            ),
        )
        assert payload == {"input": "Example text to embed", "model": "provider-id", "truncate": True}


class TestPublicAIProvider:
    def test_prepare_url(self):
        helper = PublicAIConversationalTask()
        assert (
            helper._prepare_url("publicai_token", "username/repo_name")
            == "https://api.publicai.co/v1/chat/completions"
        )


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


class TestOVHcloudAIEndpointsProvider:
    def test_prepare_hf_url_conversational(self):
        helper = OVHcloudConversationalTask()
        url = helper._prepare_url("hf_token", "username/repo_name")
        assert url == "https://router.huggingface.co/ovhcloud/v1/chat/completions"

    def test_prepare_url_conversational(self):
        helper = OVHcloudConversationalTask()
        url = helper._prepare_url("ovhcloud_token", "username/repo_name")
        assert url == "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/chat/completions"

    def test_prepare_payload_as_dict(self):
        helper = OVHcloudConversationalTask()
        payload = helper._prepare_payload_as_dict(
            [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello!"},
            ],
            {
                "max_tokens": 512,
                "temperature": 0.15,
                "top_p": 1,
                "presence_penalty": 0,
                "stream": True,
            },
            InferenceProviderMapping(
                provider="ovhcloud",
                hf_model_id="meta-llama/Llama-3.1-8B-Instruct",
                providerId="Llama-3.1-8B-Instruct",
                task="conversational",
                status="live",
            ),
        )
        assert payload == {
            "max_tokens": 512,
            "messages": [
                {"content": "You are a helpful assistant", "role": "system"},
                {"role": "user", "content": "Hello!"},
            ],
            "model": "Llama-3.1-8B-Instruct",
            "presence_penalty": 0,
            "stream": True,
            "temperature": 0.15,
            "top_p": 1,
        }

    def test_prepare_route_conversational(self):
        helper = OVHcloudConversationalTask()
        assert helper._prepare_route("username/repo_name", "hf_token") == "/v1/chat/completions"


class TestReplicateProvider:
    def test_automatic_speech_recognition_payload(self):
        helper = ReplicateAutomaticSpeechRecognitionTask()

        mapping_info = InferenceProviderMapping(
            provider="replicate",
            hf_model_id="openai/whisper-large-v3",
            providerId="openai/whisper-large-v3",
            task="automatic-speech-recognition",
            status="live",
        )

        payload = helper._prepare_payload_as_dict(
            "https://example.com/audio.mp3",
            {"language": "en"},
            mapping_info,
        )

        assert payload == {"input": {"audio": "https://example.com/audio.mp3", "language": "en"}}

        mapping_with_version = InferenceProviderMapping(
            provider="replicate",
            hf_model_id="openai/whisper-large-v3",
            providerId="openai/whisper-large-v3:123",
            task="automatic-speech-recognition",
            status="live",
        )

        audio_bytes = b"dummy-audio"
        encoded_audio = base64.b64encode(audio_bytes).decode()

        payload = helper._prepare_payload_as_dict(
            audio_bytes,
            {},
            mapping_with_version,
        )

        assert payload == {
            "input": {"audio": f"data:audio/wav;base64,{encoded_audio}"},
            "version": "123",
        }

    def test_automatic_speech_recognition_get_response_variants(self, mocker):
        helper = ReplicateAutomaticSpeechRecognitionTask()

        result = helper.get_response({"output": "hello"})
        assert result == {"text": "hello"}

        result = helper.get_response({"output": ["hello-world"]})
        assert result == {"text": "hello-world"}

        result = helper.get_response({"output": {"transcription": "bonjour"}})
        assert result == {"text": "bonjour"}

        result = helper.get_response({"output": {"translation": "hola"}})
        assert result == {"text": "hola"}

        mock_session = mocker.patch("huggingface_hub.inference._providers.replicate.get_session")
        mock_response = mocker.Mock(text="file text")
        mock_response.raise_for_status = lambda: None
        mock_session.return_value.get.return_value = mock_response

        result = helper.get_response({"output": {"txt_file": "https://example.com/output.txt"}})
        mock_session.return_value.get.assert_called_once_with("https://example.com/output.txt")
        assert result == {"text": "file text"}

        with pytest.raises(ValueError):
            helper.get_response({"output": 123})

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


class TestWavespeedAIProvider:
    """Test Wavespeed AI provider functionality."""

    def test_prepare_headers(self):
        """Test header preparation for both direct and routed calls."""
        helper = WavespeedAITextToImageTask()

        # Test with Wavespeed API key
        headers = helper._prepare_headers({}, "ws_test_key")
        assert headers["authorization"] == "Bearer ws_test_key"

        # Test with HF token
        headers = helper._prepare_headers({}, "hf_token")
        assert headers["authorization"] == "Bearer hf_token"

    def test_prepare_text_to_image_payload(self):
        """Test payload preparation for text-to-image task."""
        helper = WavespeedAITextToImageTask()
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
                provider="wavespeed",
                hf_model_id="black-forest-labs/FLUX.1-schnell",
                providerId="wavespeed-ai/flux-schnell",
                task="text-to-image",
                status="live",
            ),
        )
        assert payload == {
            "prompt": "a beautiful cat",
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512,
            "seed": 42,
        }

    def test_prepare_text_to_video_payload(self):
        """Test payload preparation for text-to-video task."""
        helper = WavespeedAITextToVideoTask()
        payload = helper._prepare_payload_as_dict(
            "a dancing cat",
            {
                "guidance_scale": 5,
                "num_inference_steps": 30,
                "seed": -1,
                "duration": 5,
                "enable_safety_checker": True,
                "flow_shift": 2.9,
                "size": "480*832",
            },
            InferenceProviderMapping(
                provider="wavespeed",
                hf_model_id="Wan-AI/Wan2.1-T2V-14B",
                providerId="wavespeed-ai/wan-2.1/t2v-480p",
                task="text-to-video",
                status="live",
            ),
        )
        assert payload == {
            "prompt": "a dancing cat",
            "guidance_scale": 5,
            "num_inference_steps": 30,
            "seed": -1,
            "duration": 5,
            "enable_safety_checker": True,
            "flow_shift": 2.9,
            "size": "480*832",
        }

    def test_prepare_image_to_image_payload(self, mocker):
        """Test payload preparation for image-to-image task."""
        helper = WavespeedAIImageToImageTask()

        # Mock image data
        image_data = b"dummy_image_data"
        mock_encode = mocker.patch("base64.b64encode")
        mock_encode.return_value.decode.return_value = "base64_encoded_image"

        payload = helper._prepare_payload_as_dict(
            image_data,
            {"prompt": "The leopard chases its prey", "guidance_scale": 5, "num_inference_steps": 30, "seed": -1},
            InferenceProviderMapping(
                provider="wavespeed",
                hf_model_id="HiDream-ai/HiDream-E1-Full",
                providerId="wavespeed-ai/hidream-e1-full",
                task="image-to-image",
                status="live",
            ),
        )

        assert payload == {
            "image": "data:image/jpeg;base64,base64_encoded_image",
            "prompt": "The leopard chases its prey",
            "guidance_scale": 5,
            "num_inference_steps": 30,
            "seed": -1,
        }
        mock_encode.assert_called_once_with(image_data)

    def test_prepare_image_to_video_payload(self, mocker):
        """Test payload preparation for image-to-video task."""
        helper = WavespeedAIImageToVideoTask()

        # Mock image data
        image_data = b"dummy_image_data"
        mock_encode = mocker.patch("base64.b64encode")
        mock_encode.return_value.decode.return_value = "base64_encoded_image"

        payload = helper._prepare_payload_as_dict(
            image_data,
            {"prompt": "The leopard chases its prey", "guidance_scale": 5, "num_inference_steps": 30, "seed": -1},
            InferenceProviderMapping(
                provider="wavespeed",
                hf_model_id="Wan-AI/Wan2.1-I2V-14B-480P",
                providerId="wavespeed-ai/wan-2.1/i2v-480p",
                task="image-to-video",
                status="live",
            ),
        )

        assert payload == {
            "image": "data:image/jpeg;base64,base64_encoded_image",
            "prompt": "The leopard chases its prey",
            "guidance_scale": 5,
            "num_inference_steps": 30,
            "seed": -1,
        }
        mock_encode.assert_called_once_with(image_data)

    def test_prepare_urls(self):
        """Test URL preparation for different tasks."""
        # Text to Image
        t2i_helper = WavespeedAITextToImageTask()
        t2i_url = t2i_helper._prepare_url("ws_test_key", "wavespeed-ai/flux-schnell")
        assert t2i_url == "https://api.wavespeed.ai/api/v3/wavespeed-ai/flux-schnell"

        # Text to Video
        t2v_helper = WavespeedAITextToVideoTask()
        t2v_url = t2v_helper._prepare_url("ws_test_key", "wavespeed-ai/wan-2.1/t2v-480p")
        assert t2v_url == "https://api.wavespeed.ai/api/v3/wavespeed-ai/wan-2.1/t2v-480p"

        # Image to Image
        i2i_helper = WavespeedAIImageToImageTask()
        i2i_url = i2i_helper._prepare_url("ws_test_key", "wavespeed-ai/hidream-e1-full")
        assert i2i_url == "https://api.wavespeed.ai/api/v3/wavespeed-ai/hidream-e1-full"

        # Image to Video
        i2v_helper = WavespeedAIImageToVideoTask()
        i2v_url = i2v_helper._prepare_url("ws_test_key", "wavespeed-ai/wan-2.1/i2v-480p")
        assert i2v_url == "https://api.wavespeed.ai/api/v3/wavespeed-ai/wan-2.1/i2v-480p"


class TestZaiProvider:
    def test_prepare_route(self):
        helper = ZaiConversationalTask()
        route = helper._prepare_route("test-model", "zai_token")
        assert route == "/api/paas/v4/chat/completions"

    def test_prepare_headers(self):
        helper = ZaiConversationalTask()
        headers = helper._prepare_headers({}, "test_key")
        assert headers["Accept-Language"] == "en-US,en"

    def test_prepare_url(self):
        helper = ZaiConversationalTask()
        assert helper.task == "conversational"
        url = helper._prepare_url("zai_token", "test-model")
        assert url == "https://api.z.ai/api/paas/v4/chat/completions"

        # Test with HF token (should route through HF proxy)
        url = helper._prepare_url("hf_token", "test-model")
        assert url.startswith("https://router.huggingface.co/zai-org")

    def test_text_to_image_prepare_route(self):
        helper = ZaiTextToImageTask()
        route = helper._prepare_route("glm-image", "zai_token")
        assert route == "/api/paas/v4/async/images/generations"

    def test_text_to_image_prepare_headers(self):
        helper = ZaiTextToImageTask()
        headers = helper._prepare_headers({}, "test_key")
        assert headers["Accept-Language"] == "en-US,en"
        assert headers["x-source-channel"] == "hugging_face"

    def test_text_to_image_prepare_url(self):
        helper = ZaiTextToImageTask()
        assert helper.task == "text-to-image"
        url = helper._prepare_url("zai_token", "glm-image")
        assert url == "https://api.z.ai/api/paas/v4/async/images/generations"

        # Test with HF token (should route through HF proxy)
        url = helper._prepare_url("hf_token", "glm-image")
        assert url.startswith("https://router.huggingface.co/zai-org")

    def test_text_to_image_prepare_payload(self):
        helper = ZaiTextToImageTask()
        payload = helper._prepare_payload_as_dict(
            "A cute cat sitting on a sunny windowsill",
            {"width": 1280, "height": 1280},
            InferenceProviderMapping(
                provider="zai-org",
                hf_model_id="zai-org/glm-image",
                providerId="glm-image",
                task="text-to-image",
                status="live",
            ),
        )
        assert payload == {
            "model": "glm-image",
            "prompt": "A cute cat sitting on a sunny windowsill",
            "size": "1280x1280",
        }

    def test_text_to_image_prepare_payload_no_size(self):
        helper = ZaiTextToImageTask()
        payload = helper._prepare_payload_as_dict(
            "A cute cat",
            {},
            InferenceProviderMapping(
                provider="zai-org",
                hf_model_id="zai-org/glm-image",
                providerId="glm-image",
                task="text-to-image",
                status="live",
            ),
        )
        assert payload == {
            "model": "glm-image",
            "prompt": "A cute cat",
        }

    def test_text_to_image_get_response_success(self, mocker):
        helper = ZaiTextToImageTask()
        mock_session = mocker.patch("huggingface_hub.inference._providers.zai_org.get_session")
        mock_sleep = mocker.patch("huggingface_hub.inference._providers.zai_org.time.sleep")

        # Mock polling response and image download
        mock_session.return_value.get.side_effect = [
            # First call: poll for status (still processing)
            mocker.Mock(
                json=lambda: {"task_status": "PROCESSING", "id": "8353992347972780031"},
                raise_for_status=lambda: None,
            ),
            # Second call: poll for status (success)
            mocker.Mock(
                json=lambda: {
                    "task_status": "SUCCESS",
                    "id": "8353992347972780031",
                    "image_result": [{"url": "https://example.com/image.png"}],
                },
                raise_for_status=lambda: None,
            ),
            # Third call: download image
            mocker.Mock(content=b"image_bytes", raise_for_status=lambda: None),
        ]

        api_key = helper._prepare_api_key("hf_token")
        headers = helper._prepare_headers({}, api_key)
        url = helper._prepare_url(api_key, "glm-image")

        request_params = RequestParameters(
            url=url,
            headers=headers,
            task="text-to-image",
            model="glm-image",
            data=None,
            json=None,
        )

        response = helper.get_response(
            {"id": "8353992347972780031", "task_status": "PROCESSING", "model": "glm-image"},
            request_params,
        )

        assert response == b"image_bytes"
        assert mock_session.return_value.get.call_count == 3
        mock_sleep.assert_called_once_with(ZAI_POLLING_INTERVAL)

    def test_text_to_image_get_response_immediate_success(self, mocker):
        """Test when the response is already successful (no polling needed)."""
        helper = ZaiTextToImageTask()
        mock_session = mocker.patch("huggingface_hub.inference._providers.zai_org.get_session")

        mock_session.return_value.get.return_value = mocker.Mock(content=b"image_bytes", raise_for_status=lambda: None)

        response = helper.get_response(
            {
                "id": "8353992347972780031",
                "task_status": "SUCCESS",
                "image_result": [{"url": "https://example.com/image.png"}],
            },
            None,
        )

        assert response == b"image_bytes"
        mock_session.return_value.get.assert_called_once_with("https://example.com/image.png")

    def test_text_to_image_get_response_fail(self):
        helper = ZaiTextToImageTask()
        with pytest.raises(ValueError, match="ZAI image generation failed"):
            helper.get_response(
                {"id": "8353992347972780031", "task_status": "FAIL"},
                None,
            )

    def test_text_to_image_get_response_no_task_id(self):
        helper = ZaiTextToImageTask()
        with pytest.raises(ValueError, match="No task_id in response"):
            helper.get_response({}, None)

    def test_text_to_image_get_response_no_image_result(self, mocker):
        helper = ZaiTextToImageTask()
        with pytest.raises(ValueError, match="No image_result in response"):
            helper.get_response(
                {"id": "8353992347972780031", "task_status": "SUCCESS"},
                None,
            )


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
def test_recursive_merge(dict1: dict, dict2: dict, expected: dict):
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
def test_filter_none(data: dict, expected: dict):
    """Test that filter_none removes None values from nested dictionaries."""
    assert filter_none(data) == expected


def test_get_provider_helper_auto_non_conversational(mocker):
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


def test_get_provider_helper_auto_conversational():
    """Test the 'auto' provider selection logic for conversational task.

    In practice, no HTTP call is made to the Hub because routing is done server-side.
    """
    helper = get_provider_helper(provider="auto", task="conversational", model="test-model")

    assert isinstance(helper, AutoRouterConversationalTask)
