import base64
import logging
from unittest.mock import MagicMock, patch

import pytest
from pytest import LogCaptureFixture

from huggingface_hub import constants
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
from huggingface_hub.inference._providers.cohere import CohereConversationalTask
from huggingface_hub.inference._providers.deepinfra import (
    DeepInfraAutomaticSpeechRecognitionTask,
    DeepInfraFeatureExtractionTask,
    DeepInfraTextToSpeechTask,
)
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
from huggingface_hub.inference._providers.scaleway import ScalewayConversationalTask, ScalewayFeatureExtractionTask
from huggingface_hub.inference._providers.together import (
    TogetherConversationalTask,
    TogetherFeatureExtractionTask,
    TogetherImageToImageTask,
    TogetherImageToVideoTask,
    TogetherTextToImageTask,
    TogetherTextToSpeechTask,
    TogetherTextToVideoTask,
)
from huggingface_hub.inference._providers.wavespeed import (
    WavespeedAIImageToImageTask,
    WavespeedAIImageToVideoTask,
    WavespeedAITextToImageTask,
    WavespeedAITextToVideoTask,
)
from huggingface_hub.inference._providers.zai_org import _POLLING_INTERVAL as ZAI_POLLING_INTERVAL
from huggingface_hub.inference._providers.zai_org import ZaiConversationalTask, ZaiTextToImageTask


pytestmark = pytest.mark.inference
