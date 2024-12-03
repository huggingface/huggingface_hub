# coding=utf-8
# Copyright 2023-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# WARNING
# This entire file has been adapted from the sync-client code in `src/huggingface_hub/inference/_client.py`.
# Any change in InferenceClient will be automatically reflected in AsyncInferenceClient.
# To re-generate the code, run `make style` or `python ./utils/generate_async_inference_client.py --update`.
# WARNING
import asyncio
import base64
import logging
import re
import time
import warnings
from typing import TYPE_CHECKING, Any, AsyncIterable, Dict, List, Literal, Optional, Set, Union, overload

from requests.structures import CaseInsensitiveDict

from huggingface_hub.constants import ALL_INFERENCE_API_FRAMEWORKS, INFERENCE_ENDPOINT, MAIN_INFERENCE_API_FRAMEWORKS
from huggingface_hub.errors import InferenceTimeoutError
from huggingface_hub.inference._common import (
    TASKS_EXPECTING_IMAGES,
    ContentT,
    ModelStatus,
    _async_stream_chat_completion_response,
    _async_stream_text_generation_response,
    _b64_encode,
    _b64_to_image,
    _bytes_to_dict,
    _bytes_to_image,
    _bytes_to_list,
    _fetch_recommended_models,
    _get_unsupported_text_generation_kwargs,
    _import_numpy,
    _open_as_binary,
    _prepare_payload,
    _set_unsupported_text_generation_kwargs,
    raise_text_generation_error,
)
from huggingface_hub.inference._generated.types import (
    AudioClassificationOutputElement,
    AudioClassificationOutputTransform,
    AudioToAudioOutputElement,
    AutomaticSpeechRecognitionOutput,
    ChatCompletionInputGrammarType,
    ChatCompletionInputStreamOptions,
    ChatCompletionInputTool,
    ChatCompletionInputToolChoiceClass,
    ChatCompletionInputToolChoiceEnum,
    ChatCompletionOutput,
    ChatCompletionStreamOutput,
    DocumentQuestionAnsweringOutputElement,
    FillMaskOutputElement,
    ImageClassificationOutputElement,
    ImageClassificationOutputTransform,
    ImageSegmentationOutputElement,
    ImageSegmentationSubtask,
    ImageToImageTargetSize,
    ImageToTextOutput,
    ObjectDetectionOutputElement,
    Padding,
    QuestionAnsweringOutputElement,
    SummarizationOutput,
    SummarizationTruncationStrategy,
    TableQuestionAnsweringOutputElement,
    TextClassificationOutputElement,
    TextClassificationOutputTransform,
    TextGenerationInputGrammarType,
    TextGenerationOutput,
    TextGenerationStreamOutput,
    TextToImageTargetSize,
    TextToSpeechEarlyStoppingEnum,
    TokenClassificationAggregationStrategy,
    TokenClassificationOutputElement,
    TranslationOutput,
    TranslationTruncationStrategy,
    VisualQuestionAnsweringOutputElement,
    ZeroShotClassificationOutputElement,
    ZeroShotImageClassificationOutputElement,
)
from huggingface_hub.utils import build_hf_headers
from huggingface_hub.utils._deprecation import _deprecate_arguments

from .._common import _async_yield_from, _import_aiohttp


if TYPE_CHECKING:
    import numpy as np
    from aiohttp import ClientResponse, ClientSession
    from PIL.Image import Image

logger = logging.getLogger(__name__)


MODEL_KWARGS_NOT_USED_REGEX = re.compile(r"The following `model_kwargs` are not used by the model: \[(.*?)\]")


class AsyncInferenceClient:
    """
    Initialize a new Inference Client.

    [`InferenceClient`] aims to provide a unified experience to perform inference. The client can be used
    seamlessly with either the (free) Inference API or self-hosted Inference Endpoints.

    Args:
        model (`str`, `optional`):
            The model to run inference with. Can be a model id hosted on the Hugging Face Hub, e.g. `meta-llama/Meta-Llama-3-8B-Instruct`
            or a URL to a deployed Inference Endpoint. Defaults to None, in which case a recommended model is
            automatically selected for the task.
            Note: for better compatibility with OpenAI's client, `model` has been aliased as `base_url`. Those 2
            arguments are mutually exclusive. If using `base_url` for chat completion, the `/chat/completions` suffix
            path will be appended to the base URL (see the [TGI Messages API](https://huggingface.co/docs/text-generation-inference/en/messages_api)
            documentation for details). When passing a URL as `model`, the client will not append any suffix path to it.
        token (`str` or `bool`, *optional*):
            Hugging Face token. Will default to the locally saved token if not provided.
            Pass `token=False` if you don't want to send your token to the server.
            Note: for better compatibility with OpenAI's client, `token` has been aliased as `api_key`. Those 2
            arguments are mutually exclusive and have the exact same behavior.
        timeout (`float`, `optional`):
            The maximum number of seconds to wait for a response from the server. Loading a new model in Inference
            API can take up to several minutes. Defaults to None, meaning it will loop until the server is available.
        headers (`Dict[str, str]`, `optional`):
            Additional headers to send to the server. By default only the authorization and user-agent headers are sent.
            Values in this dictionary will override the default values.
        cookies (`Dict[str, str]`, `optional`):
            Additional cookies to send to the server.
        trust_env ('bool', 'optional'):
            Trust environment settings for proxy configuration if the parameter is `True` (`False` by default).
        proxies (`Any`, `optional`):
            Proxies to use for the request.
        base_url (`str`, `optional`):
            Base URL to run inference. This is a duplicated argument from `model` to make [`InferenceClient`]
            follow the same pattern as `openai.OpenAI` client. Cannot be used if `model` is set. Defaults to None.
        api_key (`str`, `optional`):
            Token to use for authentication. This is a duplicated argument from `token` to make [`InferenceClient`]
            follow the same pattern as `openai.OpenAI` client. Cannot be used if `token` is set. Defaults to None.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        token: Union[str, bool, None] = None,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        trust_env: bool = False,
        proxies: Optional[Any] = None,
        # OpenAI compatibility
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        if model is not None and base_url is not None:
            raise ValueError(
                "Received both `model` and `base_url` arguments. Please provide only one of them."
                " `base_url` is an alias for `model` to make the API compatible with OpenAI's client."
                " If using `base_url` for chat completion, the `/chat/completions` suffix path will be appended to the base url."
                " When passing a URL as `model`, the client will not append any suffix path to it."
            )
        if token is not None and api_key is not None:
            raise ValueError(
                "Received both `token` and `api_key` arguments. Please provide only one of them."
                " `api_key` is an alias for `token` to make the API compatible with OpenAI's client."
                " It has the exact same behavior as `token`."
            )

        self.model: Optional[str] = model
        self.token: Union[str, bool, None] = token if token is not None else api_key
        self.headers: CaseInsensitiveDict[str] = CaseInsensitiveDict(
            build_hf_headers(token=self.token)  # 'authorization' + 'user-agent'
        )
        if headers is not None:
            self.headers.update(headers)
        self.cookies = cookies
        self.timeout = timeout
        self.trust_env = trust_env
        self.proxies = proxies

        # OpenAI compatibility
        self.base_url = base_url

        # Keep track of the sessions to close them properly
        self._sessions: Dict["ClientSession", Set["ClientResponse"]] = dict()

    def __repr__(self):
        return f"<InferenceClient(model='{self.model if self.model else ''}', timeout={self.timeout})>"

    @overload
    async def post(  # type: ignore[misc]
        self,
        *,
        json: Optional[Union[str, Dict, List]] = None,
        data: Optional[ContentT] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        stream: Literal[False] = ...,
    ) -> bytes: ...

    @overload
    async def post(  # type: ignore[misc]
        self,
        *,
        json: Optional[Union[str, Dict, List]] = None,
        data: Optional[ContentT] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        stream: Literal[True] = ...,
    ) -> AsyncIterable[bytes]: ...

    @overload
    async def post(
        self,
        *,
        json: Optional[Union[str, Dict, List]] = None,
        data: Optional[ContentT] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        stream: bool = False,
    ) -> Union[bytes, AsyncIterable[bytes]]: ...

    async def post(
        self,
        *,
        json: Optional[Union[str, Dict, List]] = None,
        data: Optional[ContentT] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        stream: bool = False,
    ) -> Union[bytes, AsyncIterable[bytes]]:
        """
        Make a POST request to the inference server.

        Args:
            json (`Union[str, Dict, List]`, *optional*):
                The JSON data to send in the request body, specific to each task. Defaults to None.
            data (`Union[str, Path, bytes, BinaryIO]`, *optional*):
                The content to send in the request body, specific to each task.
                It can be raw bytes, a pointer to an opened file, a local file path,
                or a URL to an online resource (image, audio file,...). If both `json` and `data` are passed,
                `data` will take precedence. At least `json` or `data` must be provided. Defaults to None.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. Will override the model defined at the instance level. Defaults to None.
            task (`str`, *optional*):
                The task to perform on the inference. All available tasks can be found
                [here](https://huggingface.co/tasks). Used only to default to a recommended model if `model` is not
                provided. At least `model` or `task` must be provided. Defaults to None.
            stream (`bool`, *optional*):
                Whether to iterate over streaming APIs.

        Returns:
            bytes: The raw bytes returned by the server.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.
        """

        aiohttp = _import_aiohttp()

        url = self._resolve_url(model, task)

        if data is not None and json is not None:
            warnings.warn("Ignoring `json` as `data` is passed as binary.")

        # Set Accept header if relevant
        headers = dict()
        if task in TASKS_EXPECTING_IMAGES and "Accept" not in headers:
            headers["Accept"] = "image/png"

        t0 = time.time()
        timeout = self.timeout
        while True:
            with _open_as_binary(data) as data_as_binary:
                # Do not use context manager as we don't want to close the connection immediately when returning
                # a stream
                session = self._get_client_session(headers=headers)

                try:
                    response = await session.post(url, json=json, data=data_as_binary, proxy=self.proxies)
                    response_error_payload = None
                    if response.status != 200:
                        try:
                            response_error_payload = await response.json()  # get payload before connection closed
                        except Exception:
                            pass
                    response.raise_for_status()
                    if stream:
                        return _async_yield_from(session, response)
                    else:
                        content = await response.read()
                        await session.close()
                        return content
                except asyncio.TimeoutError as error:
                    await session.close()
                    # Convert any `TimeoutError` to a `InferenceTimeoutError`
                    raise InferenceTimeoutError(f"Inference call timed out: {url}") from error  # type: ignore
                except aiohttp.ClientResponseError as error:
                    error.response_error_payload = response_error_payload
                    await session.close()
                    if response.status == 422 and task is not None:
                        error.message += f". Make sure '{task}' task is supported by the model."
                    if response.status == 503:
                        # If Model is unavailable, either raise a TimeoutError...
                        if timeout is not None and time.time() - t0 > timeout:
                            raise InferenceTimeoutError(
                                f"Model not loaded on the server: {url}. Please retry with a higher timeout"
                                f" (current: {self.timeout}).",
                                request=error.request,
                                response=error.response,
                            ) from error
                        # ...or wait 1s and retry
                        logger.info(f"Waiting for model to be loaded on the server: {error}")
                        if "X-wait-for-model" not in headers and url.startswith(INFERENCE_ENDPOINT):
                            headers["X-wait-for-model"] = "1"
                        await asyncio.sleep(1)
                        if timeout is not None:
                            timeout = max(self.timeout - (time.time() - t0), 1)  # type: ignore
                        continue
                    raise error
                except Exception:
                    await session.close()
                    raise

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    def __del__(self):
        if len(self._sessions) > 0:
            warnings.warn(
                "Deleting 'AsyncInferenceClient' client but some sessions are still open. "
                "This can happen if you've stopped streaming data from the server before the stream was complete. "
                "To close the client properly, you must call `await client.close()` "
                "or use an async context (e.g. `async with AsyncInferenceClient(): ...`."
            )

    async def close(self):
        """Close all open sessions.

        By default, 'aiohttp.ClientSession' objects are closed automatically when a call is completed. However, if you
        are streaming data from the server and you stop before the stream is complete, you must call this method to
        close the session properly.

        Another possibility is to use an async context (e.g. `async with AsyncInferenceClient(): ...`).
        """
        await asyncio.gather(*[session.close() for session in self._sessions.keys()])

    async def audio_classification(
        self,
        audio: ContentT,
        *,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        function_to_apply: Optional["AudioClassificationOutputTransform"] = None,
    ) -> List[AudioClassificationOutputElement]:
        """
        Perform audio classification on the provided audio content.

        Args:
            audio (Union[str, Path, bytes, BinaryIO]):
                The audio content to classify. It can be raw audio bytes, a local audio file, or a URL pointing to an
                audio file.
            model (`str`, *optional*):
                The model to use for audio classification. Can be a model ID hosted on the Hugging Face Hub
                or a URL to a deployed Inference Endpoint. If not provided, the default recommended model for
                audio classification will be used.
            top_k (`int`, *optional*):
                When specified, limits the output to the top K most probable classes.
            function_to_apply (`"AudioClassificationOutputTransform"`, *optional*):
                The function to apply to the model outputs in order to retrieve the scores.

        Returns:
            `List[AudioClassificationOutputElement]`: List of [`AudioClassificationOutputElement`] items containing the predicted labels and their confidence.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.audio_classification("audio.flac")
        [
            AudioClassificationOutputElement(score=0.4976358711719513, label='hap'),
            AudioClassificationOutputElement(score=0.3677836060523987, label='neu'),
            ...
        ]
        ```
        """
        parameters = {"function_to_apply": function_to_apply, "top_k": top_k}
        payload = _prepare_payload(audio, parameters=parameters, expect_binary=True)
        response = await self.post(**payload, model=model, task="audio-classification")
        return AudioClassificationOutputElement.parse_obj_as_list(response)

    async def audio_to_audio(
        self,
        audio: ContentT,
        *,
        model: Optional[str] = None,
    ) -> List[AudioToAudioOutputElement]:
        """
        Performs multiple tasks related to audio-to-audio depending on the model (eg: speech enhancement, source separation).

        Args:
            audio (Union[str, Path, bytes, BinaryIO]):
                The audio content for the model. It can be raw audio bytes, a local audio file, or a URL pointing to an
                audio file.
            model (`str`, *optional*):
                The model can be any model which takes an audio file and returns another audio file. Can be a model ID hosted on the Hugging Face Hub
                or a URL to a deployed Inference Endpoint. If not provided, the default recommended model for
                audio_to_audio will be used.

        Returns:
            `List[AudioToAudioOutputElement]`: A list of [`AudioToAudioOutputElement`] items containing audios label, content-type, and audio content in blob.

        Raises:
            `InferenceTimeoutError`:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> audio_output = await client.audio_to_audio("audio.flac")
        >>> async for i, item in enumerate(audio_output):
        >>>     with open(f"output_{i}.flac", "wb") as f:
                    f.write(item.blob)
        ```
        """
        response = await self.post(data=audio, model=model, task="audio-to-audio")
        audio_output = AudioToAudioOutputElement.parse_obj_as_list(response)
        for item in audio_output:
            item.blob = base64.b64decode(item.blob)
        return audio_output

    async def automatic_speech_recognition(
        self,
        audio: ContentT,
        *,
        model: Optional[str] = None,
    ) -> AutomaticSpeechRecognitionOutput:
        """
        Perform automatic speech recognition (ASR or audio-to-text) on the given audio content.

        Args:
            audio (Union[str, Path, bytes, BinaryIO]):
                The content to transcribe. It can be raw audio bytes, local audio file, or a URL to an audio file.
            model (`str`, *optional*):
                The model to use for ASR. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. If not provided, the default recommended model for ASR will be used.

        Returns:
            [`AutomaticSpeechRecognitionOutput`]: An item containing the transcribed text and optionally the timestamp chunks.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.automatic_speech_recognition("hello_world.flac").text
        "hello world"
        ```
        """
        response = await self.post(data=audio, model=model, task="automatic-speech-recognition")
        return AutomaticSpeechRecognitionOutput.parse_obj_as_instance(response)

    @overload
    async def chat_completion(  # type: ignore
        self,
        messages: List[Dict],
        *,
        model: Optional[str] = None,
        stream: Literal[False] = False,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[List[float]] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatCompletionInputGrammarType] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream_options: Optional[ChatCompletionInputStreamOptions] = None,
        temperature: Optional[float] = None,
        tool_choice: Optional[Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"]] = None,
        tool_prompt: Optional[str] = None,
        tools: Optional[List[ChatCompletionInputTool]] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> ChatCompletionOutput: ...

    @overload
    async def chat_completion(  # type: ignore
        self,
        messages: List[Dict],
        *,
        model: Optional[str] = None,
        stream: Literal[True] = True,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[List[float]] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatCompletionInputGrammarType] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream_options: Optional[ChatCompletionInputStreamOptions] = None,
        temperature: Optional[float] = None,
        tool_choice: Optional[Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"]] = None,
        tool_prompt: Optional[str] = None,
        tools: Optional[List[ChatCompletionInputTool]] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> AsyncIterable[ChatCompletionStreamOutput]: ...

    @overload
    async def chat_completion(
        self,
        messages: List[Dict],
        *,
        model: Optional[str] = None,
        stream: bool = False,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[List[float]] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatCompletionInputGrammarType] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream_options: Optional[ChatCompletionInputStreamOptions] = None,
        temperature: Optional[float] = None,
        tool_choice: Optional[Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"]] = None,
        tool_prompt: Optional[str] = None,
        tools: Optional[List[ChatCompletionInputTool]] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Union[ChatCompletionOutput, AsyncIterable[ChatCompletionStreamOutput]]: ...

    async def chat_completion(
        self,
        messages: List[Dict],
        *,
        model: Optional[str] = None,
        stream: bool = False,
        # Parameters from ChatCompletionInput (handled manually)
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[List[float]] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatCompletionInputGrammarType] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream_options: Optional[ChatCompletionInputStreamOptions] = None,
        temperature: Optional[float] = None,
        tool_choice: Optional[Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"]] = None,
        tool_prompt: Optional[str] = None,
        tools: Optional[List[ChatCompletionInputTool]] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Union[ChatCompletionOutput, AsyncIterable[ChatCompletionStreamOutput]]:
        """
        A method for completing conversations using a specified language model.

        <Tip>

        The `client.chat_completion` method is aliased as `client.chat.completions.create` for compatibility with OpenAI's client.
        Inputs and outputs are strictly the same and using either syntax will yield the same results.
        Check out the [Inference guide](https://huggingface.co/docs/huggingface_hub/guides/inference#openai-compatibility)
        for more details about OpenAI's compatibility.

        </Tip>

        Args:
            messages (List of [`ChatCompletionInputMessage`]):
                Conversation history consisting of roles and content pairs.
            model (`str`, *optional*):
                The model to use for chat-completion. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. If not provided, the default recommended model for chat-based text-generation will be used.
                See https://huggingface.co/tasks/text-generation for more details.

                If `model` is a model ID, it is passed to the server as the `model` parameter. If you want to define a
                custom URL while setting `model` in the request payload, you must set `base_url` when initializing [`InferenceClient`].
            frequency_penalty (`float`, *optional*):
                Penalizes new tokens based on their existing frequency
                in the text so far. Range: [-2.0, 2.0]. Defaults to 0.0.
            logit_bias (`List[float]`, *optional*):
                Modify the likelihood of specified tokens appearing in the completion. Accepts a JSON object that maps tokens
                (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically,
                the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model,
                but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should
                result in a ban or exclusive selection of the relevant token. Defaults to None.
            logprobs (`bool`, *optional*):
                Whether to return log probabilities of the output tokens or not. If true, returns the log
                probabilities of each output token returned in the content of message.
            max_tokens (`int`, *optional*):
                Maximum number of tokens allowed in the response. Defaults to 100.
            n (`int`, *optional*):
                UNUSED.
            presence_penalty (`float`, *optional*):
                Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the
                text so far, increasing the model's likelihood to talk about new topics.
            response_format ([`ChatCompletionInputGrammarType`], *optional*):
                Grammar constraints. Can be either a JSONSchema or a regex.
            seed (Optional[`int`], *optional*):
                Seed for reproducible control flow. Defaults to None.
            stop (Optional[`str`], *optional*):
                Up to four strings which trigger the end of the response.
                Defaults to None.
            stream (`bool`, *optional*):
                Enable realtime streaming of responses. Defaults to False.
            stream_options ([`ChatCompletionInputStreamOptions`], *optional*):
                Options for streaming completions.
            temperature (`float`, *optional*):
                Controls randomness of the generations. Lower values ensure
                less random completions. Range: [0, 2]. Defaults to 1.0.
            top_logprobs (`int`, *optional*):
                An integer between 0 and 5 specifying the number of most likely tokens to return at each token
                position, each with an associated log probability. logprobs must be set to true if this parameter is
                used.
            top_p (`float`, *optional*):
                Fraction of the most likely next words to sample from.
                Must be between 0 and 1. Defaults to 1.0.
            tool_choice ([`ChatCompletionInputToolChoiceClass`] or [`ChatCompletionInputToolChoiceEnum`], *optional*):
                The tool to use for the completion. Defaults to "auto".
            tool_prompt (`str`, *optional*):
                A prompt to be appended before the tools.
            tools (List of [`ChatCompletionInputTool`], *optional*):
                A list of tools the model may call. Currently, only functions are supported as a tool. Use this to
                provide a list of functions the model may generate JSON inputs for.

        Returns:
            [`ChatCompletionOutput`] or Iterable of [`ChatCompletionStreamOutput`]:
            Generated text returned from the server:
            - if `stream=False`, the generated text is returned as a [`ChatCompletionOutput`] (default).
            - if `stream=True`, the generated text is returned token by token as a sequence of [`ChatCompletionStreamOutput`].

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:

        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> messages = [{"role": "user", "content": "What is the capital of France?"}]
        >>> client = AsyncInferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")
        >>> await client.chat_completion(messages, max_tokens=100)
        ChatCompletionOutput(
            choices=[
                ChatCompletionOutputComplete(
                    finish_reason='eos_token',
                    index=0,
                    message=ChatCompletionOutputMessage(
                        role='assistant',
                        content='The capital of France is Paris.',
                        name=None,
                        tool_calls=None
                    ),
                    logprobs=None
                )
            ],
            created=1719907176,
            id='',
            model='meta-llama/Meta-Llama-3-8B-Instruct',
            object='text_completion',
            system_fingerprint='2.0.4-sha-f426a33',
            usage=ChatCompletionOutputUsage(
                completion_tokens=8,
                prompt_tokens=17,
                total_tokens=25
            )
        )
        ```

        Example using streaming:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> messages = [{"role": "user", "content": "What is the capital of France?"}]
        >>> client = AsyncInferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")
        >>> async for token in await client.chat_completion(messages, max_tokens=10, stream=True):
        ...     print(token)
        ChatCompletionStreamOutput(choices=[ChatCompletionStreamOutputChoice(delta=ChatCompletionStreamOutputDelta(content='The', role='assistant'), index=0, finish_reason=None)], created=1710498504)
        ChatCompletionStreamOutput(choices=[ChatCompletionStreamOutputChoice(delta=ChatCompletionStreamOutputDelta(content=' capital', role='assistant'), index=0, finish_reason=None)], created=1710498504)
        (...)
        ChatCompletionStreamOutput(choices=[ChatCompletionStreamOutputChoice(delta=ChatCompletionStreamOutputDelta(content=' may', role='assistant'), index=0, finish_reason=None)], created=1710498504)
        ```

        Example using OpenAI's syntax:
        ```py
        # Must be run in an async context
        # instead of `from openai import OpenAI`
        from huggingface_hub import AsyncInferenceClient

        # instead of `client = OpenAI(...)`
        client = AsyncInferenceClient(
            base_url=...,
            api_key=...,
        )

        output = await client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count to 10"},
            ],
            stream=True,
            max_tokens=1024,
        )

        for chunk in output:
            print(chunk.choices[0].delta.content)
        ```

        Example using Image + Text as input:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient

        # provide a remote URL
        >>> image_url ="https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        # or a base64-encoded image
        >>> image_path = "/path/to/image.jpeg"
        >>> with open(image_path, "rb") as f:
        ...     base64_image = base64.b64encode(f.read()).decode("utf-8")
        >>> image_url = f"data:image/jpeg;base64,{base64_image}"

        >>> client = AsyncInferenceClient("meta-llama/Llama-3.2-11B-Vision-Instruct")
        >>> output = await client.chat.completions.create(
        ...     messages=[
        ...         {
        ...             "role": "user",
        ...             "content": [
        ...                 {
        ...                     "type": "image_url",
        ...                     "image_url": {"url": image_url},
        ...                 },
        ...                 {
        ...                     "type": "text",
        ...                     "text": "Describe this image in one sentence.",
        ...                 },
        ...             ],
        ...         },
        ...     ],
        ... )
        >>> output
        The image depicts the iconic Statue of Liberty situated in New York Harbor, New York, on a clear day.
        ```

        Example using tools:
        ```py
        # Must be run in an async context
        >>> client = AsyncInferenceClient("meta-llama/Meta-Llama-3-70B-Instruct")
        >>> messages = [
        ...     {
        ...         "role": "system",
        ...         "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
        ...     },
        ...     {
        ...         "role": "user",
        ...         "content": "What's the weather like the next 3 days in San Francisco, CA?",
        ...     },
        ... ]
        >>> tools = [
        ...     {
        ...         "type": "function",
        ...         "function": {
        ...             "name": "get_current_weather",
        ...             "description": "Get the current weather",
        ...             "parameters": {
        ...                 "type": "object",
        ...                 "properties": {
        ...                     "location": {
        ...                         "type": "string",
        ...                         "description": "The city and state, e.g. San Francisco, CA",
        ...                     },
        ...                     "format": {
        ...                         "type": "string",
        ...                         "enum": ["celsius", "fahrenheit"],
        ...                         "description": "The temperature unit to use. Infer this from the users location.",
        ...                     },
        ...                 },
        ...                 "required": ["location", "format"],
        ...             },
        ...         },
        ...     },
        ...     {
        ...         "type": "function",
        ...         "function": {
        ...             "name": "get_n_day_weather_forecast",
        ...             "description": "Get an N-day weather forecast",
        ...             "parameters": {
        ...                 "type": "object",
        ...                 "properties": {
        ...                     "location": {
        ...                         "type": "string",
        ...                         "description": "The city and state, e.g. San Francisco, CA",
        ...                     },
        ...                     "format": {
        ...                         "type": "string",
        ...                         "enum": ["celsius", "fahrenheit"],
        ...                         "description": "The temperature unit to use. Infer this from the users location.",
        ...                     },
        ...                     "num_days": {
        ...                         "type": "integer",
        ...                         "description": "The number of days to forecast",
        ...                     },
        ...                 },
        ...                 "required": ["location", "format", "num_days"],
        ...             },
        ...         },
        ...     },
        ... ]

        >>> response = await client.chat_completion(
        ...     model="meta-llama/Meta-Llama-3-70B-Instruct",
        ...     messages=messages,
        ...     tools=tools,
        ...     tool_choice="auto",
        ...     max_tokens=500,
        ... )
        >>> response.choices[0].message.tool_calls[0].function
        ChatCompletionOutputFunctionDefinition(
            arguments={
                'location': 'San Francisco, CA',
                'format': 'fahrenheit',
                'num_days': 3
            },
            name='get_n_day_weather_forecast',
            description=None
        )
        ```

        Example using response_format:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient("meta-llama/Meta-Llama-3-70B-Instruct")
        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": "I saw a puppy a cat and a raccoon during my bike ride in the park. What did I saw and when?",
        ...     },
        ... ]
        >>> response_format = {
        ...     "type": "json",
        ...     "value": {
        ...         "properties": {
        ...             "location": {"type": "string"},
        ...             "activity": {"type": "string"},
        ...             "animals_seen": {"type": "integer", "minimum": 1, "maximum": 5},
        ...             "animals": {"type": "array", "items": {"type": "string"}},
        ...         },
        ...         "required": ["location", "activity", "animals_seen", "animals"],
        ...     },
        ... }
        >>> response = await client.chat_completion(
        ...     messages=messages,
        ...     response_format=response_format,
        ...     max_tokens=500,
        )
        >>> response.choices[0].message.content
        '{\n\n"activity": "bike ride",\n"animals": ["puppy", "cat", "raccoon"],\n"animals_seen": 3,\n"location": "park"}'
        ```
        """
        model_url = self._resolve_chat_completion_url(model)

        # `model` is sent in the payload. Not used by the server but can be useful for debugging/routing.
        # If it's a ID on the Hub => use it. Otherwise, we use a random string.
        model_id = model or self.model or "tgi"
        if model_id.startswith(("http://", "https://")):
            model_id = "tgi"  # dummy value

        payload = dict(
            model=model_id,
            messages=messages,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            temperature=temperature,
            tool_choice=tool_choice,
            tool_prompt=tool_prompt,
            tools=tools,
            top_logprobs=top_logprobs,
            top_p=top_p,
            stream=stream,
            stream_options=stream_options,
        )
        payload = {key: value for key, value in payload.items() if value is not None}
        data = await self.post(model=model_url, json=payload, stream=stream)

        if stream:
            return _async_stream_chat_completion_response(data)  # type: ignore[arg-type]

        return ChatCompletionOutput.parse_obj_as_instance(data)  # type: ignore[arg-type]

    def _resolve_chat_completion_url(self, model: Optional[str] = None) -> str:
        # Since `chat_completion(..., model=xxx)` is also a payload parameter for the server, we need to handle 'model' differently.
        # `self.base_url` and `self.model` takes precedence over 'model' argument only in `chat_completion`.
        model_id_or_url = self.base_url or self.model or model or self.get_recommended_model("text-generation")

        # Resolve URL if it's a model ID
        model_url = (
            model_id_or_url
            if model_id_or_url.startswith(("http://", "https://"))
            else self._resolve_url(model_id_or_url, task="text-generation")
        )

        # Strip trailing /
        model_url = model_url.rstrip("/")

        # Append /chat/completions if not already present
        if model_url.endswith("/v1"):
            model_url += "/chat/completions"

        # Append /v1/chat/completions if not already present
        if not model_url.endswith("/chat/completions"):
            model_url += "/v1/chat/completions"

        return model_url

    async def document_question_answering(
        self,
        image: ContentT,
        question: str,
        *,
        model: Optional[str] = None,
        doc_stride: Optional[int] = None,
        handle_impossible_answer: Optional[bool] = None,
        lang: Optional[str] = None,
        max_answer_len: Optional[int] = None,
        max_question_len: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        top_k: Optional[int] = None,
        word_boxes: Optional[List[Union[List[float], str]]] = None,
    ) -> List[DocumentQuestionAnsweringOutputElement]:
        """
        Answer questions on document images.

        Args:
            image (`Union[str, Path, bytes, BinaryIO]`):
                The input image for the context. It can be raw bytes, an image file, or a URL to an online image.
            question (`str`):
                Question to be answered.
            model (`str`, *optional*):
                The model to use for the document question answering task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended document question answering model will be used.
                Defaults to None.
            doc_stride (`int`, *optional*):
                If the words in the document are too long to fit with the question for the model, it will be split in
                several chunks with some overlap. This argument controls the size of that overlap.
            handle_impossible_answer (`bool`, *optional*):
                Whether to accept impossible as an answer
            lang (`str`, *optional*):
                Language to use while running OCR. Defaults to english.
            max_answer_len (`int`, *optional*):
                The maximum length of predicted answers (e.g., only answers with a shorter length are considered).
            max_question_len (`int`, *optional*):
                The maximum length of the question after tokenization. It will be truncated if needed.
            max_seq_len (`int`, *optional*):
                The maximum length of the total sentence (context + question) in tokens of each chunk passed to the
                model. The context will be split in several chunks (using doc_stride as overlap) if needed.
            top_k (`int`, *optional*):
                The number of answers to return (will be chosen by order of likelihood). Can return less than top_k
                answers if there are not enough options available within the context.
            word_boxes (`List[Union[List[float], str`, *optional*):
                A list of words and bounding boxes (normalized 0->1000). If provided, the inference will skip the OCR
                step and use the provided bounding boxes instead.
        Returns:
            `List[DocumentQuestionAnsweringOutputElement]`: a list of [`DocumentQuestionAnsweringOutputElement`] items containing the predicted label, associated probability, word ids, and page number.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.


        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.document_question_answering(image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png", question="What is the invoice number?")
        [DocumentQuestionAnsweringOutputElement(answer='us-001', end=16, score=0.9999666213989258, start=16)]
        ```
        """
        inputs: Dict[str, Any] = {"question": question, "image": _b64_encode(image)}
        parameters = {
            "doc_stride": doc_stride,
            "handle_impossible_answer": handle_impossible_answer,
            "lang": lang,
            "max_answer_len": max_answer_len,
            "max_question_len": max_question_len,
            "max_seq_len": max_seq_len,
            "top_k": top_k,
            "word_boxes": word_boxes,
        }
        payload = _prepare_payload(inputs, parameters=parameters)
        response = await self.post(**payload, model=model, task="document-question-answering")
        return DocumentQuestionAnsweringOutputElement.parse_obj_as_list(response)

    async def feature_extraction(
        self,
        text: str,
        *,
        normalize: Optional[bool] = None,
        prompt_name: Optional[str] = None,
        truncate: Optional[bool] = None,
        truncation_direction: Optional[Literal["Left", "Right"]] = None,
        model: Optional[str] = None,
    ) -> "np.ndarray":
        """
        Generate embeddings for a given text.

        Args:
            text (`str`):
                The text to embed.
            model (`str`, *optional*):
                The model to use for the conversational task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended conversational model will be used.
                Defaults to None.
            normalize (`bool`, *optional*):
                Whether to normalize the embeddings or not.
                Only available on server powered by Text-Embedding-Inference.
            prompt_name (`str`, *optional*):
                The name of the prompt that should be used by for encoding. If not set, no prompt will be applied.
                Must be a key in the `Sentence Transformers` configuration `prompts` dictionary.
                For example if ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ",...},
                then the sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"
                because the prompt text will be prepended before any text to encode.
            truncate (`bool`, *optional*):
                Whether to truncate the embeddings or not.
                Only available on server powered by Text-Embedding-Inference.
            truncation_direction (`Literal["Left", "Right"]`, *optional*):
                Which side of the input should be truncated when `truncate=True` is passed.

        Returns:
            `np.ndarray`: The embedding representing the input text as a float32 numpy array.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.feature_extraction("Hi, who are you?")
        array([[ 2.424802  ,  2.93384   ,  1.1750331 , ...,  1.240499, -0.13776633, -0.7889173 ],
        [-0.42943227, -0.6364878 , -1.693462  , ...,  0.41978157, -2.4336355 ,  0.6162071 ],
        ...,
        [ 0.28552425, -0.928395  , -1.2077185 , ...,  0.76810825, -2.1069427 ,  0.6236161 ]], dtype=float32)
        ```
        """
        parameters = {
            "normalize": normalize,
            "prompt_name": prompt_name,
            "truncate": truncate,
            "truncation_direction": truncation_direction,
        }
        payload = _prepare_payload(text, parameters=parameters)
        response = await self.post(**payload, model=model, task="feature-extraction")
        np = _import_numpy()
        return np.array(_bytes_to_dict(response), dtype="float32")

    async def fill_mask(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        targets: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[FillMaskOutputElement]:
        """
        Fill in a hole with a missing word (token to be precise).

        Args:
            text (`str`):
                a string to be filled from, must contain the [MASK] token (check model card for exact name of the mask).
            model (`str`, *optional*):
                The model to use for the fill mask task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended fill mask model will be used.
            targets (`List[str`, *optional*):
                When passed, the model will limit the scores to the passed targets instead of looking up in the whole
                vocabulary. If the provided targets are not in the model vocab, they will be tokenized and the first
                resulting token will be used (with a warning, and that might be slower).
            top_k (`int`, *optional*):
                When passed, overrides the number of predictions to return.
        Returns:
            `List[FillMaskOutputElement]`: a list of [`FillMaskOutputElement`] items containing the predicted label, associated
            probability, token reference, and completed text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.fill_mask("The goal of life is <mask>.")
        [
            FillMaskOutputElement(score=0.06897063553333282, token=11098, token_str=' happiness', sequence='The goal of life is happiness.'),
            FillMaskOutputElement(score=0.06554922461509705, token=45075, token_str=' immortality', sequence='The goal of life is immortality.')
        ]
        ```
        """
        parameters = {"targets": targets, "top_k": top_k}
        payload = _prepare_payload(text, parameters=parameters)
        response = await self.post(**payload, model=model, task="fill-mask")
        return FillMaskOutputElement.parse_obj_as_list(response)

    async def image_classification(
        self,
        image: ContentT,
        *,
        model: Optional[str] = None,
        function_to_apply: Optional["ImageClassificationOutputTransform"] = None,
        top_k: Optional[int] = None,
    ) -> List[ImageClassificationOutputElement]:
        """
        Perform image classification on the given image using the specified model.

        Args:
            image (`Union[str, Path, bytes, BinaryIO]`):
                The image to classify. It can be raw bytes, an image file, or a URL to an online image.
            model (`str`, *optional*):
                The model to use for image classification. Can be a model ID hosted on the Hugging Face Hub or a URL to a
                deployed Inference Endpoint. If not provided, the default recommended model for image classification will be used.
            function_to_apply (`"ImageClassificationOutputTransform"`, *optional*):
                The function to apply to the model outputs in order to retrieve the scores.
            top_k (`int`, *optional*):
                When specified, limits the output to the top K most probable classes.
        Returns:
            `List[ImageClassificationOutputElement]`: a list of [`ImageClassificationOutputElement`] items containing the predicted label and associated probability.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.image_classification("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
        [ImageClassificationOutputElement(label='Blenheim spaniel', score=0.9779096841812134), ...]
        ```
        """
        parameters = {"function_to_apply": function_to_apply, "top_k": top_k}
        payload = _prepare_payload(image, parameters=parameters, expect_binary=True)
        response = await self.post(**payload, model=model, task="image-classification")
        return ImageClassificationOutputElement.parse_obj_as_list(response)

    async def image_segmentation(
        self,
        image: ContentT,
        *,
        model: Optional[str] = None,
        mask_threshold: Optional[float] = None,
        overlap_mask_area_threshold: Optional[float] = None,
        subtask: Optional["ImageSegmentationSubtask"] = None,
        threshold: Optional[float] = None,
    ) -> List[ImageSegmentationOutputElement]:
        """
        Perform image segmentation on the given image using the specified model.

        <Tip warning={true}>

        You must have `PIL` installed if you want to work with images (`pip install Pillow`).

        </Tip>

        Args:
            image (`Union[str, Path, bytes, BinaryIO]`):
                The image to segment. It can be raw bytes, an image file, or a URL to an online image.
            model (`str`, *optional*):
                The model to use for image segmentation. Can be a model ID hosted on the Hugging Face Hub or a URL to a
                deployed Inference Endpoint. If not provided, the default recommended model for image segmentation will be used.
            mask_threshold (`float`, *optional*):
                Threshold to use when turning the predicted masks into binary values.
            overlap_mask_area_threshold (`float`, *optional*):
                Mask overlap threshold to eliminate small, disconnected segments.
            subtask (`"ImageSegmentationSubtask"`, *optional*):
                Segmentation task to be performed, depending on model capabilities.
            threshold (`float`, *optional*):
                Probability threshold to filter out predicted masks.
        Returns:
            `List[ImageSegmentationOutputElement]`: A list of [`ImageSegmentationOutputElement`] items containing the segmented masks and associated attributes.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.image_segmentation("cat.jpg")
        [ImageSegmentationOutputElement(score=0.989008, label='LABEL_184', mask=<PIL.PngImagePlugin.PngImageFile image mode=L size=400x300 at 0x7FDD2B129CC0>), ...]
        ```
        """
        parameters = {
            "mask_threshold": mask_threshold,
            "overlap_mask_area_threshold": overlap_mask_area_threshold,
            "subtask": subtask,
            "threshold": threshold,
        }
        payload = _prepare_payload(image, parameters=parameters, expect_binary=True)
        response = await self.post(**payload, model=model, task="image-segmentation")
        output = ImageSegmentationOutputElement.parse_obj_as_list(response)
        for item in output:
            item.mask = _b64_to_image(item.mask)  # type: ignore [assignment]
        return output

    async def image_to_image(
        self,
        image: ContentT,
        prompt: Optional[str] = None,
        *,
        negative_prompt: Optional[List[str]] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        model: Optional[str] = None,
        target_size: Optional[ImageToImageTargetSize] = None,
        **kwargs,
    ) -> "Image":
        """
        Perform image-to-image translation using a specified model.

        <Tip warning={true}>

        You must have `PIL` installed if you want to work with images (`pip install Pillow`).

        </Tip>

        Args:
            image (`Union[str, Path, bytes, BinaryIO]`):
                The input image for translation. It can be raw bytes, an image file, or a URL to an online image.
            prompt (`str`, *optional*):
                The text prompt to guide the image generation.
            negative_prompt (`List[str]`, *optional*):
                One or several prompt to guide what NOT to include in image generation.
            num_inference_steps (`int`, *optional*):
                For diffusion models. The number of denoising steps. More denoising steps usually lead to a higher
                quality image at the expense of slower inference.
            guidance_scale (`float`, *optional*):
                For diffusion models. A higher guidance scale value encourages the model to generate images closely
                linked to the text prompt at the expense of lower image quality.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.
            target_size (`ImageToImageTargetSize`, *optional*):
                The size in pixel of the output image.

        Returns:
            `Image`: The translated image.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> image = await client.image_to_image("cat.jpg", prompt="turn the cat into a tiger")
        >>> image.save("tiger.jpg")
        ```
        """
        parameters = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "target_size": target_size,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            **kwargs,
        }
        payload = _prepare_payload(image, parameters=parameters, expect_binary=True)
        response = await self.post(**payload, model=model, task="image-to-image")
        return _bytes_to_image(response)

    async def image_to_text(self, image: ContentT, *, model: Optional[str] = None) -> ImageToTextOutput:
        """
        Takes an input image and return text.

        Models can have very different outputs depending on your use case (image captioning, optical character recognition
        (OCR), Pix2Struct, etc). Please have a look to the model card to learn more about a model's specificities.

        Args:
            image (`Union[str, Path, bytes, BinaryIO]`):
                The input image to caption. It can be raw bytes, an image file, or a URL to an online image..
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.

        Returns:
            [`ImageToTextOutput`]: The generated text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.image_to_text("cat.jpg")
        'a cat standing in a grassy field '
        >>> await client.image_to_text("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
        'a dog laying on the grass next to a flower pot '
        ```
        """
        response = await self.post(data=image, model=model, task="image-to-text")
        output = ImageToTextOutput.parse_obj(response)
        return output[0] if isinstance(output, list) else output

    async def list_deployed_models(
        self, frameworks: Union[None, str, Literal["all"], List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        List models deployed on the Serverless Inference API service.

        This helper checks deployed models framework by framework. By default, it will check the 4 main frameworks that
        are supported and account for 95% of the hosted models. However, if you want a complete list of models you can
        specify `frameworks="all"` as input. Alternatively, if you know before-hand which framework you are interested
        in, you can also restrict to search to this one (e.g. `frameworks="text-generation-inference"`). The more
        frameworks are checked, the more time it will take.

        <Tip warning={true}>

        This endpoint method does not return a live list of all models available for the Serverless Inference API service.
        It searches over a cached list of models that were recently available and the list may not be up to date.
        If you want to know the live status of a specific model, use [`~InferenceClient.get_model_status`].

        </Tip>

        <Tip>

        This endpoint method is mostly useful for discoverability. If you already know which model you want to use and want to
        check its availability, you can directly use [`~InferenceClient.get_model_status`].

        </Tip>

        Args:
            frameworks (`Literal["all"]` or `List[str]` or `str`, *optional*):
                The frameworks to filter on. By default only a subset of the available frameworks are tested. If set to
                "all", all available frameworks will be tested. It is also possible to provide a single framework or a
                custom set of frameworks to check.

        Returns:
            `Dict[str, List[str]]`: A dictionary mapping task names to a sorted list of model IDs.

        Example:
        ```py
        # Must be run in an async contextthon
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()

        # Discover zero-shot-classification models currently deployed
        >>> models = await client.list_deployed_models()
        >>> models["zero-shot-classification"]
        ['Narsil/deberta-large-mnli-zero-cls', 'facebook/bart-large-mnli', ...]

        # List from only 1 framework
        >>> await client.list_deployed_models("text-generation-inference")
        {'text-generation': ['bigcode/starcoder', 'meta-llama/Llama-2-70b-chat-hf', ...], ...}
        ```
        """
        # Resolve which frameworks to check
        if frameworks is None:
            frameworks = MAIN_INFERENCE_API_FRAMEWORKS
        elif frameworks == "all":
            frameworks = ALL_INFERENCE_API_FRAMEWORKS
        elif isinstance(frameworks, str):
            frameworks = [frameworks]
        frameworks = list(set(frameworks))

        # Fetch them iteratively
        models_by_task: Dict[str, List[str]] = {}

        def _unpack_response(framework: str, items: List[Dict]) -> None:
            for model in items:
                if framework == "sentence-transformers":
                    # Model running with the `sentence-transformers` framework can work with both tasks even if not
                    # branded as such in the API response
                    models_by_task.setdefault("feature-extraction", []).append(model["model_id"])
                    models_by_task.setdefault("sentence-similarity", []).append(model["model_id"])
                else:
                    models_by_task.setdefault(model["task"], []).append(model["model_id"])

        async def _fetch_framework(framework: str) -> None:
            async with self._get_client_session() as client:
                response = await client.get(f"{INFERENCE_ENDPOINT}/framework/{framework}", proxy=self.proxies)
                response.raise_for_status()
                _unpack_response(framework, await response.json())

        import asyncio

        await asyncio.gather(*[_fetch_framework(framework) for framework in frameworks])

        # Sort alphabetically for discoverability and return
        for task, models in models_by_task.items():
            models_by_task[task] = sorted(set(models), key=lambda x: x.lower())
        return models_by_task

    async def object_detection(
        self, image: ContentT, *, model: Optional[str] = None, threshold: Optional[float] = None
    ) -> List[ObjectDetectionOutputElement]:
        """
        Perform object detection on the given image using the specified model.

        <Tip warning={true}>

        You must have `PIL` installed if you want to work with images (`pip install Pillow`).

        </Tip>

        Args:
            image (`Union[str, Path, bytes, BinaryIO]`):
                The image to detect objects on. It can be raw bytes, an image file, or a URL to an online image.
            model (`str`, *optional*):
                The model to use for object detection. Can be a model ID hosted on the Hugging Face Hub or a URL to a
                deployed Inference Endpoint. If not provided, the default recommended model for object detection (DETR) will be used.
            threshold (`float`, *optional*):
                The probability necessary to make a prediction.
        Returns:
            `List[ObjectDetectionOutputElement]`: A list of [`ObjectDetectionOutputElement`] items containing the bounding boxes and associated attributes.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.
            `ValueError`:
                If the request output is not a List.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.object_detection("people.jpg")
        [ObjectDetectionOutputElement(score=0.9486683011054993, label='person', box=ObjectDetectionBoundingBox(xmin=59, ymin=39, xmax=420, ymax=510)), ...]
        ```
        """
        parameters = {
            "threshold": threshold,
        }
        payload = _prepare_payload(image, parameters=parameters, expect_binary=True)
        response = await self.post(**payload, model=model, task="object-detection")
        return ObjectDetectionOutputElement.parse_obj_as_list(response)

    async def question_answering(
        self,
        question: str,
        context: str,
        *,
        model: Optional[str] = None,
        align_to_words: Optional[bool] = None,
        doc_stride: Optional[int] = None,
        handle_impossible_answer: Optional[bool] = None,
        max_answer_len: Optional[int] = None,
        max_question_len: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> Union[QuestionAnsweringOutputElement, List[QuestionAnsweringOutputElement]]:
        """
        Retrieve the answer to a question from a given text.

        Args:
            question (`str`):
                Question to be answered.
            context (`str`):
                The context of the question.
            model (`str`):
                The model to use for the question answering task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint.
            align_to_words (`bool`, *optional*):
                Attempts to align the answer to real words. Improves quality on space separated languages. Might hurt
                on non-space-separated languages (like Japanese or Chinese)
            doc_stride (`int`, *optional*):
                If the context is too long to fit with the question for the model, it will be split in several chunks
                with some overlap. This argument controls the size of that overlap.
            handle_impossible_answer (`bool`, *optional*):
                Whether to accept impossible as an answer.
            max_answer_len (`int`, *optional*):
                The maximum length of predicted answers (e.g., only answers with a shorter length are considered).
            max_question_len (`int`, *optional*):
                The maximum length of the question after tokenization. It will be truncated if needed.
            max_seq_len (`int`, *optional*):
                The maximum length of the total sentence (context + question) in tokens of each chunk passed to the
                model. The context will be split in several chunks (using docStride as overlap) if needed.
            top_k (`int`, *optional*):
                The number of answers to return (will be chosen by order of likelihood). Note that we return less than
                topk answers if there are not enough options available within the context.

        Returns:
            Union[`QuestionAnsweringOutputElement`, List[`QuestionAnsweringOutputElement`]]:
                When top_k is 1 or not provided, it returns a single `QuestionAnsweringOutputElement`.
                When top_k is greater than 1, it returns a list of `QuestionAnsweringOutputElement`.
        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.question_answering(question="What's my name?", context="My name is Clara and I live in Berkeley.")
        QuestionAnsweringOutputElement(answer='Clara', end=16, score=0.9326565265655518, start=11)
        ```
        """
        parameters = {
            "align_to_words": align_to_words,
            "doc_stride": doc_stride,
            "handle_impossible_answer": handle_impossible_answer,
            "max_answer_len": max_answer_len,
            "max_question_len": max_question_len,
            "max_seq_len": max_seq_len,
            "top_k": top_k,
        }
        inputs: Dict[str, Any] = {"question": question, "context": context}
        payload = _prepare_payload(inputs, parameters=parameters)
        response = await self.post(
            **payload,
            model=model,
            task="question-answering",
        )
        # Parse the response as a single `QuestionAnsweringOutputElement` when top_k is 1 or not provided, or a list of `QuestionAnsweringOutputElement` to ensure backward compatibility.
        output = QuestionAnsweringOutputElement.parse_obj(response)
        return output

    async def sentence_similarity(
        self, sentence: str, other_sentences: List[str], *, model: Optional[str] = None
    ) -> List[float]:
        """
        Compute the semantic similarity between a sentence and a list of other sentences by comparing their embeddings.

        Args:
            sentence (`str`):
                The main sentence to compare to others.
            other_sentences (`List[str]`):
                The list of sentences to compare to.
            model (`str`, *optional*):
                The model to use for the conversational task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended conversational model will be used.
                Defaults to None.

        Returns:
            `List[float]`: The embedding representing the input text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.sentence_similarity(
        ...     "Machine learning is so easy.",
        ...     other_sentences=[
        ...         "Deep learning is so straightforward.",
        ...         "This is so difficult, like rocket science.",
        ...         "I can't believe how much I struggled with this.",
        ...     ],
        ... )
        [0.7785726189613342, 0.45876261591911316, 0.2906220555305481]
        ```
        """
        response = await self.post(
            json={"inputs": {"source_sentence": sentence, "sentences": other_sentences}},
            model=model,
            task="sentence-similarity",
        )
        return _bytes_to_list(response)

    @_deprecate_arguments(
        version="0.29",
        deprecated_args=["parameters"],
        custom_message=(
            "The `parameters` argument is deprecated and will be removed in a future version. "
            "Provide individual parameters instead: `clean_up_tokenization_spaces`, `generate_parameters`, and `truncation`."
        ),
    )
    async def summarization(
        self,
        text: str,
        *,
        parameters: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        clean_up_tokenization_spaces: Optional[bool] = None,
        generate_parameters: Optional[Dict[str, Any]] = None,
        truncation: Optional["SummarizationTruncationStrategy"] = None,
    ) -> SummarizationOutput:
        """
        Generate a summary of a given text using a specified model.

        Args:
            text (`str`):
                The input text to summarize.
            parameters (`Dict[str, Any]`, *optional*):
                Additional parameters for summarization. Check out this [page](https://huggingface.co/docs/api-inference/detailed_parameters#summarization-task)
                for more details.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. If not provided, the default recommended model for summarization will be used.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether to clean up the potential extra spaces in the text output.
            generate_parameters (`Dict[str, Any]`, *optional*):
                Additional parametrization of the text generation algorithm.
            truncation (`"SummarizationTruncationStrategy"`, *optional*):
                The truncation strategy to use.
        Returns:
            [`SummarizationOutput`]: The generated summary text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.summarization("The Eiffel tower...")
        SummarizationOutput(generated_text="The Eiffel tower is one of the most famous landmarks in the world....")
        ```
        """
        if parameters is None:
            parameters = {
                "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
                "generate_parameters": generate_parameters,
                "truncation": truncation,
            }
        payload = _prepare_payload(text, parameters=parameters)
        response = await self.post(**payload, model=model, task="summarization")
        return SummarizationOutput.parse_obj_as_list(response)[0]

    async def table_question_answering(
        self,
        table: Dict[str, Any],
        query: str,
        *,
        model: Optional[str] = None,
        padding: Optional["Padding"] = None,
        sequential: Optional[bool] = None,
        truncation: Optional[bool] = None,
    ) -> TableQuestionAnsweringOutputElement:
        """
        Retrieve the answer to a question from information given in a table.

        Args:
            table (`str`):
                A table of data represented as a dict of lists where entries are headers and the lists are all the
                values, all lists must have the same size.
            query (`str`):
                The query in plain text that you want to ask the table.
            model (`str`):
                The model to use for the table-question-answering task. Can be a model ID hosted on the Hugging Face
                Hub or a URL to a deployed Inference Endpoint.
            padding (`"Padding"`, *optional*):
                Activates and controls padding.
            sequential (`bool`, *optional*):
                Whether to do inference sequentially or as a batch. Batching is faster, but models like SQA require the
                inference to be done sequentially to extract relations within sequences, given their conversational
                nature.
            truncation (`bool`, *optional*):
                Activates and controls truncation.

        Returns:
            [`TableQuestionAnsweringOutputElement`]: a table question answering output containing the answer, coordinates, cells and the aggregator used.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> query = "How many stars does the transformers repository have?"
        >>> table = {"Repository": ["Transformers", "Datasets", "Tokenizers"], "Stars": ["36542", "4512", "3934"]}
        >>> await client.table_question_answering(table, query, model="google/tapas-base-finetuned-wtq")
        TableQuestionAnsweringOutputElement(answer='36542', coordinates=[[0, 1]], cells=['36542'], aggregator='AVERAGE')
        ```
        """
        parameters = {
            "padding": padding,
            "sequential": sequential,
            "truncation": truncation,
        }
        inputs = {
            "query": query,
            "table": table,
        }
        payload = _prepare_payload(inputs, parameters=parameters)
        response = await self.post(
            **payload,
            model=model,
            task="table-question-answering",
        )
        return TableQuestionAnsweringOutputElement.parse_obj_as_instance(response)

    async def tabular_classification(self, table: Dict[str, Any], *, model: Optional[str] = None) -> List[str]:
        """
        Classifying a target category (a group) based on a set of attributes.

        Args:
            table (`Dict[str, Any]`):
                Set of attributes to classify.
            model (`str`, *optional*):
                The model to use for the tabular classification task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended tabular classification model will be used.
                Defaults to None.

        Returns:
            `List`: a list of labels, one per row in the initial table.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> table = {
        ...     "fixed_acidity": ["7.4", "7.8", "10.3"],
        ...     "volatile_acidity": ["0.7", "0.88", "0.32"],
        ...     "citric_acid": ["0", "0", "0.45"],
        ...     "residual_sugar": ["1.9", "2.6", "6.4"],
        ...     "chlorides": ["0.076", "0.098", "0.073"],
        ...     "free_sulfur_dioxide": ["11", "25", "5"],
        ...     "total_sulfur_dioxide": ["34", "67", "13"],
        ...     "density": ["0.9978", "0.9968", "0.9976"],
        ...     "pH": ["3.51", "3.2", "3.23"],
        ...     "sulphates": ["0.56", "0.68", "0.82"],
        ...     "alcohol": ["9.4", "9.8", "12.6"],
        ... }
        >>> await client.tabular_classification(table=table, model="julien-c/wine-quality")
        ["5", "5", "5"]
        ```
        """
        response = await self.post(
            json={"table": table},
            model=model,
            task="tabular-classification",
        )
        return _bytes_to_list(response)

    async def tabular_regression(self, table: Dict[str, Any], *, model: Optional[str] = None) -> List[float]:
        """
        Predicting a numerical target value given a set of attributes/features in a table.

        Args:
            table (`Dict[str, Any]`):
                Set of attributes stored in a table. The attributes used to predict the target can be both numerical and categorical.
            model (`str`, *optional*):
                The model to use for the tabular regression task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended tabular regression model will be used.
                Defaults to None.

        Returns:
            `List`: a list of predicted numerical target values.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> table = {
        ...     "Height": ["11.52", "12.48", "12.3778"],
        ...     "Length1": ["23.2", "24", "23.9"],
        ...     "Length2": ["25.4", "26.3", "26.5"],
        ...     "Length3": ["30", "31.2", "31.1"],
        ...     "Species": ["Bream", "Bream", "Bream"],
        ...     "Width": ["4.02", "4.3056", "4.6961"],
        ... }
        >>> await client.tabular_regression(table, model="scikit-learn/Fish-Weight")
        [110, 120, 130]
        ```
        """
        response = await self.post(json={"table": table}, model=model, task="tabular-regression")
        return _bytes_to_list(response)

    async def text_classification(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        function_to_apply: Optional["TextClassificationOutputTransform"] = None,
    ) -> List[TextClassificationOutputElement]:
        """
        Perform text classification (e.g. sentiment-analysis) on the given text.

        Args:
            text (`str`):
                A string to be classified.
            model (`str`, *optional*):
                The model to use for the text classification task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended text classification model will be used.
                Defaults to None.
            top_k (`int`, *optional*):
                When specified, limits the output to the top K most probable classes.
            function_to_apply (`"TextClassificationOutputTransform"`, *optional*):
                The function to apply to the model outputs in order to retrieve the scores.

        Returns:
            `List[TextClassificationOutputElement]`: a list of [`TextClassificationOutputElement`] items containing the predicted label and associated probability.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.text_classification("I like you")
        [
            TextClassificationOutputElement(label='POSITIVE', score=0.9998695850372314),
            TextClassificationOutputElement(label='NEGATIVE', score=0.0001304351753788069),
        ]
        ```
        """
        parameters = {
            "function_to_apply": function_to_apply,
            "top_k": top_k,
        }
        payload = _prepare_payload(text, parameters=parameters)
        response = await self.post(
            **payload,
            model=model,
            task="text-classification",
        )
        return TextClassificationOutputElement.parse_obj_as_list(response)[0]  # type: ignore [return-value]

    @overload
    async def text_generation(  # type: ignore
        self,
        prompt: str,
        *,
        details: Literal[False] = ...,
        stream: Literal[False] = ...,
        model: Optional[str] = None,
        # Parameters from `TextGenerationInputGenerateParameters` (maintained manually)
        adapter_id: Optional[str] = None,
        best_of: Optional[int] = None,
        decoder_input_details: Optional[bool] = None,
        do_sample: Optional[bool] = False,  # Manual default value
        frequency_penalty: Optional[float] = None,
        grammar: Optional[TextGenerationInputGrammarType] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: Optional[bool] = False,  # Manual default value
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stop_sequences: Optional[List[str]] = None,  # Deprecated, use `stop` instead
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_n_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: Optional[bool] = None,
    ) -> str: ...

    @overload
    async def text_generation(  # type: ignore
        self,
        prompt: str,
        *,
        details: Literal[True] = ...,
        stream: Literal[False] = ...,
        model: Optional[str] = None,
        # Parameters from `TextGenerationInputGenerateParameters` (maintained manually)
        adapter_id: Optional[str] = None,
        best_of: Optional[int] = None,
        decoder_input_details: Optional[bool] = None,
        do_sample: Optional[bool] = False,  # Manual default value
        frequency_penalty: Optional[float] = None,
        grammar: Optional[TextGenerationInputGrammarType] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: Optional[bool] = False,  # Manual default value
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stop_sequences: Optional[List[str]] = None,  # Deprecated, use `stop` instead
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_n_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: Optional[bool] = None,
    ) -> TextGenerationOutput: ...

    @overload
    async def text_generation(  # type: ignore
        self,
        prompt: str,
        *,
        details: Literal[False] = ...,
        stream: Literal[True] = ...,
        model: Optional[str] = None,
        # Parameters from `TextGenerationInputGenerateParameters` (maintained manually)
        adapter_id: Optional[str] = None,
        best_of: Optional[int] = None,
        decoder_input_details: Optional[bool] = None,
        do_sample: Optional[bool] = False,  # Manual default value
        frequency_penalty: Optional[float] = None,
        grammar: Optional[TextGenerationInputGrammarType] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: Optional[bool] = False,  # Manual default value
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stop_sequences: Optional[List[str]] = None,  # Deprecated, use `stop` instead
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_n_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: Optional[bool] = None,
    ) -> AsyncIterable[str]: ...

    @overload
    async def text_generation(  # type: ignore
        self,
        prompt: str,
        *,
        details: Literal[True] = ...,
        stream: Literal[True] = ...,
        model: Optional[str] = None,
        # Parameters from `TextGenerationInputGenerateParameters` (maintained manually)
        adapter_id: Optional[str] = None,
        best_of: Optional[int] = None,
        decoder_input_details: Optional[bool] = None,
        do_sample: Optional[bool] = False,  # Manual default value
        frequency_penalty: Optional[float] = None,
        grammar: Optional[TextGenerationInputGrammarType] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: Optional[bool] = False,  # Manual default value
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stop_sequences: Optional[List[str]] = None,  # Deprecated, use `stop` instead
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_n_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: Optional[bool] = None,
    ) -> AsyncIterable[TextGenerationStreamOutput]: ...

    @overload
    async def text_generation(
        self,
        prompt: str,
        *,
        details: Literal[True] = ...,
        stream: bool = ...,
        model: Optional[str] = None,
        # Parameters from `TextGenerationInputGenerateParameters` (maintained manually)
        adapter_id: Optional[str] = None,
        best_of: Optional[int] = None,
        decoder_input_details: Optional[bool] = None,
        do_sample: Optional[bool] = False,  # Manual default value
        frequency_penalty: Optional[float] = None,
        grammar: Optional[TextGenerationInputGrammarType] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: Optional[bool] = False,  # Manual default value
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stop_sequences: Optional[List[str]] = None,  # Deprecated, use `stop` instead
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_n_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: Optional[bool] = None,
    ) -> Union[TextGenerationOutput, AsyncIterable[TextGenerationStreamOutput]]: ...

    async def text_generation(
        self,
        prompt: str,
        *,
        details: bool = False,
        stream: bool = False,
        model: Optional[str] = None,
        # Parameters from `TextGenerationInputGenerateParameters` (maintained manually)
        adapter_id: Optional[str] = None,
        best_of: Optional[int] = None,
        decoder_input_details: Optional[bool] = None,
        do_sample: Optional[bool] = False,  # Manual default value
        frequency_penalty: Optional[float] = None,
        grammar: Optional[TextGenerationInputGrammarType] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: Optional[bool] = False,  # Manual default value
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stop_sequences: Optional[List[str]] = None,  # Deprecated, use `stop` instead
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_n_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: Optional[bool] = None,
    ) -> Union[str, TextGenerationOutput, AsyncIterable[str], AsyncIterable[TextGenerationStreamOutput]]:
        """
        Given a prompt, generate the following text.

        API endpoint is supposed to run with the `text-generation-inference` backend (TGI). This backend is the
        go-to solution to run large language models at scale. However, for some smaller models (e.g. "gpt2") the
        default `transformers` + `api-inference` solution is still in use. Both approaches have very similar APIs, but
        not exactly the same. This method is compatible with both approaches but some parameters are only available for
        `text-generation-inference`. If some parameters are ignored, a warning message is triggered but the process
        continues correctly.

        To learn more about the TGI project, please refer to https://github.com/huggingface/text-generation-inference.

        <Tip>

        If you want to generate a response from chat messages, you should use the [`InferenceClient.chat_completion`] method.
        It accepts a list of messages instead of a single text prompt and handles the chat templating for you.

        </Tip>

        Args:
            prompt (`str`):
                Input text.
            details (`bool`, *optional*):
                By default, text_generation returns a string. Pass `details=True` if you want a detailed output (tokens,
                probabilities, seed, finish reason, etc.). Only available for models running on with the
                `text-generation-inference` backend.
            stream (`bool`, *optional*):
                By default, text_generation returns the full generated text. Pass `stream=True` if you want a stream of
                tokens to be returned. Only available for models running on with the `text-generation-inference`
                backend.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.
            adapter_id (`str`, *optional*):
                Lora adapter id.
            best_of (`int`, *optional*):
                Generate best_of sequences and return the one if the highest token logprobs.
            decoder_input_details (`bool`, *optional*):
                Return the decoder input token logprobs and ids. You must set `details=True` as well for it to be taken
                into account. Defaults to `False`.
            do_sample (`bool`, *optional*):
                Activate logits sampling
            frequency_penalty (`float`, *optional*):
                Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in
                the text so far, decreasing the model's likelihood to repeat the same line verbatim.
            grammar ([`TextGenerationInputGrammarType`], *optional*):
                Grammar constraints. Can be either a JSONSchema or a regex.
            max_new_tokens (`int`, *optional*):
                Maximum number of generated tokens. Defaults to 100.
            repetition_penalty (`float`, *optional*):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            return_full_text (`bool`, *optional*):
                Whether to prepend the prompt to the generated text
            seed (`int`, *optional*):
                Random sampling seed
            stop (`List[str]`, *optional*):
                Stop generating tokens if a member of `stop` is generated.
            stop_sequences (`List[str]`, *optional*):
                Deprecated argument. Use `stop` instead.
            temperature (`float`, *optional*):
                The value used to module the logits distribution.
            top_n_tokens (`int`, *optional*):
                Return information about the `top_n_tokens` most likely tokens at each generation step, instead of
                just the sampled token.
            top_k (`int`, *optional`):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`, *optional`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            truncate (`int`, *optional`):
                Truncate inputs tokens to the given size.
            typical_p (`float`, *optional`):
                Typical Decoding mass
                See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
            watermark (`bool`, *optional`):
                Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)

        Returns:
            `Union[str, TextGenerationOutput, Iterable[str], Iterable[TextGenerationStreamOutput]]`:
            Generated text returned from the server:
            - if `stream=False` and `details=False`, the generated text is returned as a `str` (default)
            - if `stream=True` and `details=False`, the generated text is returned token by token as a `Iterable[str]`
            - if `stream=False` and `details=True`, the generated text is returned with more details as a [`~huggingface_hub.TextGenerationOutput`]
            - if `details=True` and `stream=True`, the generated text is returned token by token as a iterable of [`~huggingface_hub.TextGenerationStreamOutput`]

        Raises:
            `ValidationError`:
                If input values are not valid. No HTTP call is made to the server.
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()

        # Case 1: generate text
        >>> await client.text_generation("The huggingface_hub library is ", max_new_tokens=12)
        '100% open source and built to be easy to use.'

        # Case 2: iterate over the generated tokens. Useful for large generation.
        >>> async for token in await client.text_generation("The huggingface_hub library is ", max_new_tokens=12, stream=True):
        ...     print(token)
        100
        %
        open
        source
        and
        built
        to
        be
        easy
        to
        use
        .

        # Case 3: get more details about the generation process.
        >>> await client.text_generation("The huggingface_hub library is ", max_new_tokens=12, details=True)
        TextGenerationOutput(
            generated_text='100% open source and built to be easy to use.',
            details=TextGenerationDetails(
                finish_reason='length',
                generated_tokens=12,
                seed=None,
                prefill=[
                    TextGenerationPrefillOutputToken(id=487, text='The', logprob=None),
                    TextGenerationPrefillOutputToken(id=53789, text=' hugging', logprob=-13.171875),
                    (...)
                    TextGenerationPrefillOutputToken(id=204, text=' ', logprob=-7.0390625)
                ],
                tokens=[
                    TokenElement(id=1425, text='100', logprob=-1.0175781, special=False),
                    TokenElement(id=16, text='%', logprob=-0.0463562, special=False),
                    (...)
                    TokenElement(id=25, text='.', logprob=-0.5703125, special=False)
                ],
                best_of_sequences=None
            )
        )

        # Case 4: iterate over the generated tokens with more details.
        # Last object is more complete, containing the full generated text and the finish reason.
        >>> async for details in await client.text_generation("The huggingface_hub library is ", max_new_tokens=12, details=True, stream=True):
        ...     print(details)
        ...
        TextGenerationStreamOutput(token=TokenElement(id=1425, text='100', logprob=-1.0175781, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=16, text='%', logprob=-0.0463562, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=1314, text=' open', logprob=-1.3359375, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=3178, text=' source', logprob=-0.28100586, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=273, text=' and', logprob=-0.5961914, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=3426, text=' built', logprob=-1.9423828, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=271, text=' to', logprob=-1.4121094, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=314, text=' be', logprob=-1.5224609, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=1833, text=' easy', logprob=-2.1132812, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=271, text=' to', logprob=-0.08520508, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=745, text=' use', logprob=-0.39453125, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(
            id=25,
            text='.',
            logprob=-0.5703125,
            special=False),
            generated_text='100% open source and built to be easy to use.',
            details=TextGenerationStreamOutputStreamDetails(finish_reason='length', generated_tokens=12, seed=None)
        )

        # Case 5: generate constrained output using grammar
        >>> response = await client.text_generation(
        ...     prompt="I saw a puppy a cat and a raccoon during my bike ride in the park",
        ...     model="HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
        ...     max_new_tokens=100,
        ...     repetition_penalty=1.3,
        ...     grammar={
        ...         "type": "json",
        ...         "value": {
        ...             "properties": {
        ...                 "location": {"type": "string"},
        ...                 "activity": {"type": "string"},
        ...                 "animals_seen": {"type": "integer", "minimum": 1, "maximum": 5},
        ...                 "animals": {"type": "array", "items": {"type": "string"}},
        ...             },
        ...             "required": ["location", "activity", "animals_seen", "animals"],
        ...         },
        ...     },
        ... )
        >>> json.loads(response)
        {
            "activity": "bike riding",
            "animals": ["puppy", "cat", "raccoon"],
            "animals_seen": 3,
            "location": "park"
        }
        ```
        """
        if decoder_input_details and not details:
            warnings.warn(
                "`decoder_input_details=True` has been passed to the server but `details=False` is set meaning that"
                " the output from the server will be truncated."
            )
            decoder_input_details = False

        if stop_sequences is not None:
            warnings.warn(
                "`stop_sequences` is a deprecated argument for `text_generation` task"
                " and will be removed in version '0.28.0'. Use `stop` instead.",
                FutureWarning,
            )
        if stop is None:
            stop = stop_sequences  # use deprecated arg if provided

        # Build payload
        parameters = {
            "adapter_id": adapter_id,
            "best_of": best_of,
            "decoder_input_details": decoder_input_details,
            "details": details,
            "do_sample": do_sample,
            "frequency_penalty": frequency_penalty,
            "grammar": grammar,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "return_full_text": return_full_text,
            "seed": seed,
            "stop": stop if stop is not None else [],
            "temperature": temperature,
            "top_k": top_k,
            "top_n_tokens": top_n_tokens,
            "top_p": top_p,
            "truncate": truncate,
            "typical_p": typical_p,
            "watermark": watermark,
        }
        parameters = {k: v for k, v in parameters.items() if v is not None}
        payload = {
            "inputs": prompt,
            "parameters": parameters,
            "stream": stream,
        }

        # Remove some parameters if not a TGI server
        unsupported_kwargs = _get_unsupported_text_generation_kwargs(model)
        if len(unsupported_kwargs) > 0:
            # The server does not support some parameters
            # => means it is not a TGI server
            # => remove unsupported parameters and warn the user

            ignored_parameters = []
            for key in unsupported_kwargs:
                if parameters.get(key):
                    ignored_parameters.append(key)
                parameters.pop(key, None)
            if len(ignored_parameters) > 0:
                warnings.warn(
                    "API endpoint/model for text-generation is not served via TGI. Ignoring following parameters:"
                    f" {', '.join(ignored_parameters)}.",
                    UserWarning,
                )
            if details:
                warnings.warn(
                    "API endpoint/model for text-generation is not served via TGI. Parameter `details=True` will"
                    " be ignored meaning only the generated text will be returned.",
                    UserWarning,
                )
                details = False
            if stream:
                raise ValueError(
                    "API endpoint/model for text-generation is not served via TGI. Cannot return output as a stream."
                    " Please pass `stream=False` as input."
                )

        # Handle errors separately for more precise error messages
        try:
            bytes_output = await self.post(json=payload, model=model, task="text-generation", stream=stream)  # type: ignore
        except _import_aiohttp().ClientResponseError as e:
            match = MODEL_KWARGS_NOT_USED_REGEX.search(e.response_error_payload["error"])
            if e.status == 400 and match:
                unused_params = [kwarg.strip("' ") for kwarg in match.group(1).split(",")]
                _set_unsupported_text_generation_kwargs(model, unused_params)
                return await self.text_generation(  # type: ignore
                    prompt=prompt,
                    details=details,
                    stream=stream,
                    model=model,
                    adapter_id=adapter_id,
                    best_of=best_of,
                    decoder_input_details=decoder_input_details,
                    do_sample=do_sample,
                    frequency_penalty=frequency_penalty,
                    grammar=grammar,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                    return_full_text=return_full_text,
                    seed=seed,
                    stop=stop,
                    temperature=temperature,
                    top_k=top_k,
                    top_n_tokens=top_n_tokens,
                    top_p=top_p,
                    truncate=truncate,
                    typical_p=typical_p,
                    watermark=watermark,
                )
            raise_text_generation_error(e)

        # Parse output
        if stream:
            return _async_stream_text_generation_response(bytes_output, details)  # type: ignore

        data = _bytes_to_dict(bytes_output)  # type: ignore[arg-type]

        # Data can be a single element (dict) or an iterable of dicts where we select the first element of.
        if isinstance(data, list):
            data = data[0]

        return TextGenerationOutput.parse_obj_as_instance(data) if details else data["generated_text"]

    async def text_to_image(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[List[str]] = None,
        height: Optional[float] = None,
        width: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        model: Optional[str] = None,
        scheduler: Optional[str] = None,
        target_size: Optional[TextToImageTargetSize] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> "Image":
        """
        Generate an image based on a given text using a specified model.

        <Tip warning={true}>

        You must have `PIL` installed if you want to work with images (`pip install Pillow`).

        </Tip>

        Args:
            prompt (`str`):
                The prompt to generate an image from.
            negative_prompt (`List[str`, *optional*):
                One or several prompt to guide what NOT to include in image generation.
            height (`float`, *optional*):
                The height in pixels of the image to generate.
            width (`float`, *optional*):
                The width in pixels of the image to generate.
            num_inference_steps (`int`, *optional*):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                prompt, but values too high may cause saturation and other artifacts.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. If not provided, the default recommended text-to-image model will be used.
                Defaults to None.
            scheduler (`str`, *optional*):
                Override the scheduler with a compatible one.
            target_size (`TextToImageTargetSize`, *optional*):
                The size in pixel of the output image
            seed (`int`, *optional*):
                Seed for the random number generator.

        Returns:
            `Image`: The generated image.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()

        >>> image = await client.text_to_image("An astronaut riding a horse on the moon.")
        >>> image.save("astronaut.png")

        >>> image = await client.text_to_image(
        ...     "An astronaut riding a horse on the moon.",
        ...     negative_prompt="low resolution, blurry",
        ...     model="stabilityai/stable-diffusion-2-1",
        ... )
        >>> image.save("better_astronaut.png")
        ```
        """

        parameters = {
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "scheduler": scheduler,
            "target_size": target_size,
            "seed": seed,
            **kwargs,
        }
        payload = _prepare_payload(prompt, parameters=parameters)
        response = await self.post(**payload, model=model, task="text-to-image")
        return _bytes_to_image(response)

    async def text_to_speech(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[Union[bool, "TextToSpeechEarlyStoppingEnum"]] = None,
        epsilon_cutoff: Optional[float] = None,
        eta_cutoff: Optional[float] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        min_length: Optional[int] = None,
        min_new_tokens: Optional[int] = None,
        num_beam_groups: Optional[int] = None,
        num_beams: Optional[int] = None,
        penalty_alpha: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        use_cache: Optional[bool] = None,
    ) -> bytes:
        """
        Synthesize an audio of a voice pronouncing a given text.

        Args:
            text (`str`):
                The text to synthesize.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. If not provided, the default recommended text-to-speech model will be used.
                Defaults to None.
            do_sample (`bool`, *optional*):
                Whether to use sampling instead of greedy decoding when generating new tokens.
            early_stopping (`Union[bool, "TextToSpeechEarlyStoppingEnum"]`, *optional*):
                Controls the stopping condition for beam-based methods.
            epsilon_cutoff (`float`, *optional*):
                If set to float strictly between 0 and 1, only tokens with a conditional probability greater than
                epsilon_cutoff will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on
                the size of the model. See [Truncation Sampling as Language Model
                Desmoothing](https://hf.co/papers/2210.15191) for more details.
            eta_cutoff (`float`, *optional*):
                Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to float strictly
                between 0 and 1, a token is only considered if it is greater than either eta_cutoff or sqrt(eta_cutoff)
                * exp(-entropy(softmax(next_token_logits))). The latter term is intuitively the expected next token
                probability, scaled by sqrt(eta_cutoff). In the paper, suggested values range from 3e-4 to 2e-3,
                depending on the size of the model. See [Truncation Sampling as Language Model
                Desmoothing](https://hf.co/papers/2210.15191) for more details.
            max_length (`int`, *optional*):
                The maximum length (in tokens) of the generated text, including the input.
            max_new_tokens (`int`, *optional*):
                The maximum number of tokens to generate. Takes precedence over max_length.
            min_length (`int`, *optional*):
                The minimum length (in tokens) of the generated text, including the input.
            min_new_tokens (`int`, *optional*):
                The minimum number of tokens to generate. Takes precedence over min_length.
            num_beam_groups (`int`, *optional*):
                Number of groups to divide num_beams into in order to ensure diversity among different groups of beams.
                See [this paper](https://hf.co/papers/1610.02424) for more details.
            num_beams (`int`, *optional*):
                Number of beams to use for beam search.
            penalty_alpha (`float`, *optional*):
                The value balances the model confidence and the degeneration penalty in contrastive search decoding.
            temperature (`float`, *optional*):
                The value used to modulate the next token probabilities.
            top_k (`int`, *optional*):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`, *optional*):
                If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to
                top_p or higher are kept for generation.
            typical_p (`float`, *optional*):
                Local typicality measures how similar the conditional probability of predicting a target token next is
                to the expected conditional probability of predicting a random token next, given the partial text
                already generated. If set to float < 1, the smallest set of the most locally typical tokens with
                probabilities that add up to typical_p or higher are kept for generation. See [this
                paper](https://hf.co/papers/2202.00666) for more details.
            use_cache (`bool`, *optional*):
                Whether the model should use the past last key/values attentions to speed up decoding

        Returns:
            `bytes`: The generated audio.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from pathlib import Path
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()

        >>> audio = await client.text_to_speech("Hello world")
        >>> Path("hello_world.flac").write_bytes(audio)
        ```
        """
        parameters = {
            "do_sample": do_sample,
            "early_stopping": early_stopping,
            "epsilon_cutoff": epsilon_cutoff,
            "eta_cutoff": eta_cutoff,
            "max_length": max_length,
            "max_new_tokens": max_new_tokens,
            "min_length": min_length,
            "min_new_tokens": min_new_tokens,
            "num_beam_groups": num_beam_groups,
            "num_beams": num_beams,
            "penalty_alpha": penalty_alpha,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "typical_p": typical_p,
            "use_cache": use_cache,
        }
        payload = _prepare_payload(text, parameters=parameters)
        response = await self.post(**payload, model=model, task="text-to-speech")
        return response

    async def token_classification(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        aggregation_strategy: Optional["TokenClassificationAggregationStrategy"] = None,
        ignore_labels: Optional[List[str]] = None,
        stride: Optional[int] = None,
    ) -> List[TokenClassificationOutputElement]:
        """
        Perform token classification on the given text.
        Usually used for sentence parsing, either grammatical, or Named Entity Recognition (NER) to understand keywords contained within text.

        Args:
            text (`str`):
                A string to be classified.
            model (`str`, *optional*):
                The model to use for the token classification task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended token classification model will be used.
                Defaults to None.
            aggregation_strategy (`"TokenClassificationAggregationStrategy"`, *optional*):
                The strategy used to fuse tokens based on model predictions
            ignore_labels (`List[str`, *optional*):
                A list of labels to ignore
            stride (`int`, *optional*):
                The number of overlapping tokens between chunks when splitting the input text.

        Returns:
            `List[TokenClassificationOutputElement]`: List of [`TokenClassificationOutputElement`] items containing the entity group, confidence score, word, start and end index.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.token_classification("My name is Sarah Jessica Parker but you can call me Jessica")
        [
            TokenClassificationOutputElement(
                entity_group='PER',
                score=0.9971321225166321,
                word='Sarah Jessica Parker',
                start=11,
                end=31,
            ),
            TokenClassificationOutputElement(
                entity_group='PER',
                score=0.9773476123809814,
                word='Jessica',
                start=52,
                end=59,
            )
        ]
        ```
        """

        parameters = {
            "aggregation_strategy": aggregation_strategy,
            "ignore_labels": ignore_labels,
            "stride": stride,
        }
        payload = _prepare_payload(text, parameters=parameters)
        response = await self.post(
            **payload,
            model=model,
            task="token-classification",
        )
        return TokenClassificationOutputElement.parse_obj_as_list(response)

    async def translation(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        clean_up_tokenization_spaces: Optional[bool] = None,
        truncation: Optional["TranslationTruncationStrategy"] = None,
        generate_parameters: Optional[Dict[str, Any]] = None,
    ) -> TranslationOutput:
        """
        Convert text from one language to another.

        Check out https://huggingface.co/tasks/translation for more information on how to choose the best model for
        your specific use case. Source and target languages usually depend on the model.
        However, it is possible to specify source and target languages for certain models. If you are working with one of these models,
        you can use `src_lang` and `tgt_lang` arguments to pass the relevant information.

        Args:
            text (`str`):
                A string to be translated.
            model (`str`, *optional*):
                The model to use for the translation task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended translation model will be used.
                Defaults to None.
            src_lang (`str`, *optional*):
                The source language of the text. Required for models that can translate from multiple languages.
            tgt_lang (`str`, *optional*):
                Target language to translate to. Required for models that can translate to multiple languages.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether to clean up the potential extra spaces in the text output.
            truncation (`"TranslationTruncationStrategy"`, *optional*):
                The truncation strategy to use.
            generate_parameters (`Dict[str, Any]`, *optional*):
                Additional parametrization of the text generation algorithm.

        Returns:
            [`TranslationOutput`]: The generated translated text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.
            `ValueError`:
                If only one of the `src_lang` and `tgt_lang` arguments are provided.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.translation("My name is Wolfgang and I live in Berlin")
        'Mein Name ist Wolfgang und ich lebe in Berlin.'
        >>> await client.translation("My name is Wolfgang and I live in Berlin", model="Helsinki-NLP/opus-mt-en-fr")
        TranslationOutput(translation_text='Je m'appelle Wolfgang et je vis à Berlin.')
        ```

        Specifying languages:
        ```py
        >>> client.translation("My name is Sarah Jessica Parker but you can call me Jessica", model="facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX")
        "Mon nom est Sarah Jessica Parker mais vous pouvez m'appeler Jessica"
        ```
        """
        # Throw error if only one of `src_lang` and `tgt_lang` was given
        if src_lang is not None and tgt_lang is None:
            raise ValueError("You cannot specify `src_lang` without specifying `tgt_lang`.")

        if src_lang is None and tgt_lang is not None:
            raise ValueError("You cannot specify `tgt_lang` without specifying `src_lang`.")
        parameters = {
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
            "truncation": truncation,
            "generate_parameters": generate_parameters,
        }
        payload = _prepare_payload(text, parameters=parameters)
        response = await self.post(**payload, model=model, task="translation")
        return TranslationOutput.parse_obj_as_list(response)[0]

    async def visual_question_answering(
        self,
        image: ContentT,
        question: str,
        *,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[VisualQuestionAnsweringOutputElement]:
        """
        Answering open-ended questions based on an image.

        Args:
            image (`Union[str, Path, bytes, BinaryIO]`):
                The input image for the context. It can be raw bytes, an image file, or a URL to an online image.
            question (`str`):
                Question to be answered.
            model (`str`, *optional*):
                The model to use for the visual question answering task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended visual question answering model will be used.
                Defaults to None.
            top_k (`int`, *optional*):
                The number of answers to return (will be chosen by order of likelihood). Note that we return less than
                topk answers if there are not enough options available within the context.
        Returns:
            `List[VisualQuestionAnsweringOutputElement]`: a list of [`VisualQuestionAnsweringOutputElement`] items containing the predicted label and associated probability.

        Raises:
            `InferenceTimeoutError`:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.visual_question_answering(
        ...     image="https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg",
        ...     question="What is the animal doing?"
        ... )
        [
            VisualQuestionAnsweringOutputElement(score=0.778609573841095, answer='laying down'),
            VisualQuestionAnsweringOutputElement(score=0.6957435607910156, answer='sitting'),
        ]
        ```
        """
        payload: Dict[str, Any] = {"question": question, "image": _b64_encode(image)}
        if top_k is not None:
            payload.setdefault("parameters", {})["top_k"] = top_k
        response = await self.post(json=payload, model=model, task="visual-question-answering")
        return VisualQuestionAnsweringOutputElement.parse_obj_as_list(response)

    @_deprecate_arguments(
        version="0.30.0",
        deprecated_args=["labels"],
        custom_message="`labels`has been renamed to `candidate_labels` and will be removed in huggingface_hub>=0.30.0.",
    )
    async def zero_shot_classification(
        self,
        text: str,
        # temporarily keeping it optional for backward compatibility.
        candidate_labels: List[str] = None,  # type: ignore
        *,
        multi_label: Optional[bool] = False,
        hypothesis_template: Optional[str] = None,
        model: Optional[str] = None,
        # deprecated argument
        labels: List[str] = None,  # type: ignore
    ) -> List[ZeroShotClassificationOutputElement]:
        """
        Provide as input a text and a set of candidate labels to classify the input text.

        Args:
            text (`str`):
                The input text to classify.
            candidate_labels (`List[str]`):
                The set of possible class labels to classify the text into.
            labels (`List[str]`, *optional*):
                (deprecated) List of strings. Each string is the verbalization of a possible label for the input text.
            multi_label (`bool`, *optional*):
                Whether multiple candidate labels can be true. If false, the scores are normalized such that the sum of
                the label likelihoods for each sequence is 1. If true, the labels are considered independent and
                probabilities are normalized for each candidate.
            hypothesis_template (`str`, *optional*):
                The sentence used in conjunction with `candidate_labels` to attempt the text classification by
                replacing the placeholder with the candidate labels.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. If not provided, the default recommended zero-shot classification model will be used.


        Returns:
            `List[ZeroShotClassificationOutputElement]`: List of [`ZeroShotClassificationOutputElement`] items containing the predicted labels and their confidence.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example with `multi_label=False`:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> text = (
        ...     "A new model offers an explanation for how the Galilean satellites formed around the solar system's"
        ...     "largest world. Konstantin Batygin did not set out to solve one of the solar system's most puzzling"
        ...     " mysteries when he went for a run up a hill in Nice, France."
        ... )
        >>> labels = ["space & cosmos", "scientific discovery", "microbiology", "robots", "archeology"]
        >>> await client.zero_shot_classification(text, labels)
        [
            ZeroShotClassificationOutputElement(label='scientific discovery', score=0.7961668968200684),
            ZeroShotClassificationOutputElement(label='space & cosmos', score=0.18570658564567566),
            ZeroShotClassificationOutputElement(label='microbiology', score=0.00730885099619627),
            ZeroShotClassificationOutputElement(label='archeology', score=0.006258360575884581),
            ZeroShotClassificationOutputElement(label='robots', score=0.004559356719255447),
        ]
        >>> await client.zero_shot_classification(text, labels, multi_label=True)
        [
            ZeroShotClassificationOutputElement(label='scientific discovery', score=0.9829297661781311),
            ZeroShotClassificationOutputElement(label='space & cosmos', score=0.755190908908844),
            ZeroShotClassificationOutputElement(label='microbiology', score=0.0005462635890580714),
            ZeroShotClassificationOutputElement(label='archeology', score=0.00047131875180639327),
            ZeroShotClassificationOutputElement(label='robots', score=0.00030448526376858354),
        ]
        ```

        Example with `multi_label=True` and a custom `hypothesis_template`:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.zero_shot_classification(
        ...    text="I really like our dinner and I'm very happy. I don't like the weather though.",
        ...    labels=["positive", "negative", "pessimistic", "optimistic"],
        ...    multi_label=True,
        ...    hypothesis_template="This text is {} towards the weather"
        ... )
        [
            ZeroShotClassificationOutputElement(label='negative', score=0.9231801629066467),
            ZeroShotClassificationOutputElement(label='pessimistic', score=0.8760990500450134),
            ZeroShotClassificationOutputElement(label='optimistic', score=0.0008674879791215062),
            ZeroShotClassificationOutputElement(label='positive', score=0.0005250611575320363)
        ]
        ```
        """
        # handle deprecation
        if labels is not None:
            if candidate_labels is not None:
                raise ValueError(
                    "Cannot specify both `labels` and `candidate_labels`. Use `candidate_labels` instead."
                )
            candidate_labels = labels
        elif candidate_labels is None:
            raise ValueError("Must specify `candidate_labels`")
        parameters = {
            "candidate_labels": candidate_labels,
            "multi_label": multi_label,
            "hypothesis_template": hypothesis_template,
        }
        payload = _prepare_payload(text, parameters=parameters)
        response = await self.post(
            **payload,
            task="zero-shot-classification",
            model=model,
        )
        output = _bytes_to_dict(response)
        return [
            ZeroShotClassificationOutputElement.parse_obj_as_instance({"label": label, "score": score})
            for label, score in zip(output["labels"], output["scores"])
        ]

    @_deprecate_arguments(
        version="0.30.0",
        deprecated_args=["labels"],
        custom_message="`labels`has been renamed to `candidate_labels` and will be removed in huggingface_hub>=0.30.0.",
    )
    async def zero_shot_image_classification(
        self,
        image: ContentT,
        # temporarily keeping it optional for backward compatibility.
        candidate_labels: List[str] = None,  # type: ignore
        *,
        model: Optional[str] = None,
        hypothesis_template: Optional[str] = None,
        # deprecated argument
        labels: List[str] = None,  # type: ignore
    ) -> List[ZeroShotImageClassificationOutputElement]:
        """
        Provide input image and text labels to predict text labels for the image.

        Args:
            image (`Union[str, Path, bytes, BinaryIO]`):
                The input image to caption. It can be raw bytes, an image file, or a URL to an online image.
            candidate_labels (`List[str]`):
                The candidate labels for this image
            labels (`List[str]`, *optional*):
                (deprecated) List of string possible labels. There must be at least 2 labels.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. If not provided, the default recommended zero-shot image classification model will be used.
            hypothesis_template (`str`, *optional*):
                The sentence used in conjunction with `candidate_labels` to attempt the image classification by
                replacing the placeholder with the candidate labels.

        Returns:
            `List[ZeroShotImageClassificationOutputElement]`: List of [`ZeroShotImageClassificationOutputElement`] items containing the predicted labels and their confidence.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()

        >>> await client.zero_shot_image_classification(
        ...     "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg",
        ...     labels=["dog", "cat", "horse"],
        ... )
        [ZeroShotImageClassificationOutputElement(label='dog', score=0.956),...]
        ```
        """
        # handle deprecation
        if labels is not None:
            if candidate_labels is not None:
                raise ValueError(
                    "Cannot specify both `labels` and `candidate_labels`. Use `candidate_labels` instead."
                )
            candidate_labels = labels
        elif candidate_labels is None:
            raise ValueError("Must specify `candidate_labels`")
        # Raise ValueError if input is less than 2 labels
        if len(candidate_labels) < 2:
            raise ValueError("You must specify at least 2 classes to compare.")
        parameters = {
            "candidate_labels": candidate_labels,
            "hypothesis_template": hypothesis_template,
        }
        payload = _prepare_payload(image, parameters=parameters, expect_binary=True)
        response = await self.post(
            **payload,
            model=model,
            task="zero-shot-image-classification",
        )
        return ZeroShotImageClassificationOutputElement.parse_obj_as_list(response)

    def _get_client_session(self, headers: Optional[Dict] = None) -> "ClientSession":
        aiohttp = _import_aiohttp()
        client_headers = self.headers.copy()
        if headers is not None:
            client_headers.update(headers)

        # Return a new aiohttp ClientSession with correct settings.
        session = aiohttp.ClientSession(
            headers=client_headers,
            cookies=self.cookies,
            timeout=aiohttp.ClientTimeout(self.timeout),
            trust_env=self.trust_env,
        )

        # Keep track of sessions to close them later
        self._sessions[session] = set()

        # Override the `._request` method to register responses to be closed
        session._wrapped_request = session._request

        async def _request(method, url, **kwargs):
            response = await session._wrapped_request(method, url, **kwargs)
            self._sessions[session].add(response)
            return response

        session._request = _request

        # Override the 'close' method to
        # 1. close ongoing responses
        # 2. deregister the session when closed
        session._close = session.close

        async def close_session():
            for response in self._sessions[session]:
                response.close()
            await session._close()
            self._sessions.pop(session, None)

        session.close = close_session
        return session

    def _resolve_url(self, model: Optional[str] = None, task: Optional[str] = None) -> str:
        model = model or self.model or self.base_url

        # If model is already a URL, ignore `task` and return directly
        if model is not None and (model.startswith("http://") or model.startswith("https://")):
            return model

        # # If no model but task is set => fetch the recommended one for this task
        if model is None:
            if task is None:
                raise ValueError(
                    "You must specify at least a model (repo_id or URL) or a task, either when instantiating"
                    " `InferenceClient` or when making a request."
                )
            model = self.get_recommended_model(task)
            logger.info(
                f"Using recommended model {model} for task {task}. Note that it is"
                f" encouraged to explicitly set `model='{model}'` as the recommended"
                " models list might get updated without prior notice."
            )

        # Compute InferenceAPI url
        return (
            # Feature-extraction and sentence-similarity are the only cases where we handle models with several tasks.
            f"{INFERENCE_ENDPOINT}/pipeline/{task}/{model}"
            if task in ("feature-extraction", "sentence-similarity")
            # Otherwise, we use the default endpoint
            else f"{INFERENCE_ENDPOINT}/models/{model}"
        )

    @staticmethod
    def get_recommended_model(task: str) -> str:
        """
        Get the model Hugging Face recommends for the input task.

        Args:
            task (`str`):
                The Hugging Face task to get which model Hugging Face recommends.
                All available tasks can be found [here](https://huggingface.co/tasks).

        Returns:
            `str`: Name of the model recommended for the input task.

        Raises:
            `ValueError`: If Hugging Face has no recommendation for the input task.
        """
        model = _fetch_recommended_models().get(task)
        if model is None:
            raise ValueError(
                f"Task {task} has no recommended model. Please specify a model"
                " explicitly. Visit https://huggingface.co/tasks for more info."
            )
        return model

    async def get_endpoint_info(self, *, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about the deployed endpoint.

        This endpoint is only available on endpoints powered by Text-Generation-Inference (TGI) or Text-Embedding-Inference (TEI).
        Endpoints powered by `transformers` return an empty payload.

        Args:
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.

        Returns:
            `Dict[str, Any]`: Information about the endpoint.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient("meta-llama/Meta-Llama-3-70B-Instruct")
        >>> await client.get_endpoint_info()
        {
            'model_id': 'meta-llama/Meta-Llama-3-70B-Instruct',
            'model_sha': None,
            'model_dtype': 'torch.float16',
            'model_device_type': 'cuda',
            'model_pipeline_tag': None,
            'max_concurrent_requests': 128,
            'max_best_of': 2,
            'max_stop_sequences': 4,
            'max_input_length': 8191,
            'max_total_tokens': 8192,
            'waiting_served_ratio': 0.3,
            'max_batch_total_tokens': 1259392,
            'max_waiting_tokens': 20,
            'max_batch_size': None,
            'validation_workers': 32,
            'max_client_batch_size': 4,
            'version': '2.0.2',
            'sha': 'dccab72549635c7eb5ddb17f43f0b7cdff07c214',
            'docker_label': 'sha-dccab72'
        }
        ```
        """
        model = model or self.model
        if model is None:
            raise ValueError("Model id not provided.")
        if model.startswith(("http://", "https://")):
            url = model.rstrip("/") + "/info"
        else:
            url = f"{INFERENCE_ENDPOINT}/models/{model}/info"

        async with self._get_client_session() as client:
            response = await client.get(url, proxy=self.proxies)
            response.raise_for_status()
            return await response.json()

    async def health_check(self, model: Optional[str] = None) -> bool:
        """
        Check the health of the deployed endpoint.

        Health check is only available with Inference Endpoints powered by Text-Generation-Inference (TGI) or Text-Embedding-Inference (TEI).
        For Inference API, please use [`InferenceClient.get_model_status`] instead.

        Args:
            model (`str`, *optional*):
                URL of the Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.

        Returns:
            `bool`: True if everything is working fine.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient("https://jzgu0buei5.us-east-1.aws.endpoints.huggingface.cloud")
        >>> await client.health_check()
        True
        ```
        """
        model = model or self.model
        if model is None:
            raise ValueError("Model id not provided.")
        if not model.startswith(("http://", "https://")):
            raise ValueError(
                "Model must be an Inference Endpoint URL. For serverless Inference API, please use `InferenceClient.get_model_status`."
            )
        url = model.rstrip("/") + "/health"

        async with self._get_client_session() as client:
            response = await client.get(url, proxy=self.proxies)
            return response.status == 200

    async def get_model_status(self, model: Optional[str] = None) -> ModelStatus:
        """
        Get the status of a model hosted on the Inference API.

        <Tip>

        This endpoint is mostly useful when you already know which model you want to use and want to check its
        availability. If you want to discover already deployed models, you should rather use [`~InferenceClient.list_deployed_models`].

        </Tip>

        Args:
            model (`str`, *optional*):
                Identifier of the model for witch the status gonna be checked. If model is not provided,
                the model associated with this instance of [`InferenceClient`] will be used. Only InferenceAPI service can be checked so the
                identifier cannot be a URL.


        Returns:
            [`ModelStatus`]: An instance of ModelStatus dataclass, containing information,
                         about the state of the model: load, state, compute type and framework.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.get_model_status("meta-llama/Meta-Llama-3-8B-Instruct")
        ModelStatus(loaded=True, state='Loaded', compute_type='gpu', framework='text-generation-inference')
        ```
        """
        model = model or self.model
        if model is None:
            raise ValueError("Model id not provided.")
        if model.startswith("https://"):
            raise NotImplementedError("Model status is only available for Inference API endpoints.")
        url = f"{INFERENCE_ENDPOINT}/status/{model}"

        async with self._get_client_session() as client:
            response = await client.get(url, proxy=self.proxies)
            response.raise_for_status()
            response_data = await response.json()

        if "error" in response_data:
            raise ValueError(response_data["error"])

        return ModelStatus(
            loaded=response_data["loaded"],
            state=response_data["state"],
            compute_type=response_data["compute_type"],
            framework=response_data["framework"],
        )

    @property
    def chat(self) -> "ProxyClientChat":
        return ProxyClientChat(self)


class _ProxyClient:
    """Proxy class to be able to call `client.chat.completion.create(...)` as OpenAI client."""

    def __init__(self, client: AsyncInferenceClient):
        self._client = client


class ProxyClientChat(_ProxyClient):
    """Proxy class to be able to call `client.chat.completion.create(...)` as OpenAI client."""

    @property
    def completions(self) -> "ProxyClientChatCompletions":
        return ProxyClientChatCompletions(self._client)


class ProxyClientChatCompletions(_ProxyClient):
    """Proxy class to be able to call `client.chat.completion.create(...)` as OpenAI client."""

    @property
    def create(self):
        return self._client.chat_completion
