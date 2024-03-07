from functools import lru_cache
from typing import Callable, Dict, List, Optional, Union

from minijinja import Environment
from minijinja import TemplateError as MiniJinjaTemplateError

from ..utils import HfHubHTTPError, RepositoryNotFoundError


class TemplateError(Exception):
    """Any error raised while trying to fetch or render a chat template."""


def render_chat_prompt(
    *,
    model_id: str,
    messages: List[Dict[str, str]],
    token: Union[str, bool, None] = None,
    add_generation_prompt: bool = True,
    **kwargs,
) -> str:
    """Render a chat prompt using a model's chat template.

    Args:
        model_id (`str`):
            The model id.
        messages (`List[Dict[str, str]]`):
            The list of messages to render.
        token (`str` or `bool`, *optional*):
            Hugging Face token. Will default to the locally saved token if not provided.

    Returns:
        `str`: The rendered chat prompt.

    Raises:
        `TemplateError`: If there's any issue while fetching, compiling or rendering the chat template.
    """
    template = _fetch_and_compile_template(model_id=model_id, token=token)

    try:
        return template(messages=messages, add_generation_prompt=add_generation_prompt, **kwargs)
    except MiniJinjaTemplateError as e:
        raise TemplateError(f"Error while trying to render chat prompt for model '{model_id}': {e}") from e


@lru_cache  # TODO: lru_cache for raised exceptions
def _fetch_and_compile_template(*, model_id: str, token: Union[str, bool, None]) -> Callable:
    """Fetch and compile a model's chat template.

    Method is cached to avoid fetching the same model's config multiple times.

    Args:
        model_id (`str`):
            The model id.
        token (`str` or `bool`, *optional*):
            Hugging Face token. Will default to the locally saved token if not provided.

    Returns:
        `Callable`: A callable that takes a list of messages and returns the rendered chat prompt.
    """
    from huggingface_hub.hf_api import model_info

    # 1. fetch config from API
    try:
        config = model_info(model_id, token=token).config
    except RepositoryNotFoundError as e:
        raise TemplateError(f"Cannot render chat template: model '{model_id}' not found.") from e
    except HfHubHTTPError as e:
        raise TemplateError(f"Error while trying to fetch chat template for model '{model_id}': {e}") from e

    # 2. check config validity
    if config is None:
        raise TemplateError(f"Config not found for model '{model_id}'.")
    tokenizer_config = config.get("tokenizer_config")
    if tokenizer_config is None:
        raise TemplateError(f"Tokenizer config not found for model '{model_id}'.")
    if tokenizer_config.get("chat_template") is None:
        raise TemplateError(f"Chat template not found in tokenizer_config for model '{model_id}'.")
    chat_template = tokenizer_config["chat_template"]
    print(chat_template)
    if not isinstance(chat_template, str):
        raise TemplateError(f"Chat template must be a string, not '{type(chat_template)}' (model: {model_id}).")

    special_tokens: Dict[str, Optional[str]] = {}
    for key, value in tokenizer_config.items():
        if "token" in key:
            if isinstance(value, str):
                special_tokens[key] = value
            elif isinstance(value, dict) and value.get("__type") == "AddedToken":
                special_tokens[key] = value.get("content")

    # 3. compile template and return
    env = Environment()
    try:
        env.add_template("chat_template", chat_template)
    except MiniJinjaTemplateError as e:
        raise TemplateError(f"Error while trying to compile chat template for model '{model_id}': {e}") from e
    return lambda **kwargs: env.render_template("chat_template", **kwargs, **special_tokens)
