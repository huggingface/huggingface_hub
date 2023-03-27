import atexit
import os
from functools import wraps
from typing import Awaitable, Callable, Dict, Optional

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


_global_app: Optional["WebhookApp"] = None
_is_local = os.getenv("SYSTEM") != "spaces"


class WebhookApp:
    """
    The [`WebhookApp`] class lets you create an instance of a Gradio app that can receive Huggingface webhooks.
    These webhooks can be registered using the [`~WebhookApp.add_webhook`] decorator. Webhook endpoints are added to
    the app as a POST endpoint to the FastAPI router. Once all the webhooks are registered, the `run` method has to be
    called to start the app.

    The [`WebhookApp`] is meant to be debugged locally before being deployed to a Space. Local debugging works with
    the HF Hub by opening a tunnel to your machine (using Gradio). You can protect your webhook server by setting a
    `webhook_secret`.

    It is recommended to accept [`WebhookPayload`] as the first argument of the webhook function. It is a Pydantic
    model that contains all the information about the webhook event. The data will be parsed automatically for you.

    Args:
        ui (`gradio.Blocks`, optional):
            A Gradio UI instance that will be use as the Space landing page. If None, a basic interface displaying
            information about the configured webhooks is created.
        webhook_secret (`str`, optional):
            A secret key to verify incoming webhook requests. You can set this value to any secret you want as long as
            you configure it as well in your [webhooks settings panel](https://huggingface.co/settings/webhooks). You
            can also set this value as the `WEBHOOK_SECRET` environment variable. If no secret is provided, the
            webhook endpoints are opened without any security.

    Example:

        The quickest way to define a webhook app is to use the [`hf_webhook`] decorator. Under the hood it will create
        a [`WebhookApp`] with the default UI and register the decorated function as a webhook. Multiple webhooks can
        be added in the same script. Once all the webhooks are defined, the `run` method will be called automatically.


        ```python
        from huggingface_hub import hf_webhook, WebhookPayload

        @hf_webhook
        async def trigger_training(payload: WebhookPayload):
            if payload.repo.type == "dataset" and payload.event.action == "update":
                # Trigger a training job if a dataset is updated
                ...
        ```

        If you need more control over the app, you can create a [`WebhookApp`] instance yourself and register webhooks
        as you would register FastAPI routes. The `run` method will have to be called manually to start the app.

        ```python
        import gradio as gr
        from huggingface_hub import WebhookApp, WebhookPayload

        with gr.Blocks as ui:
            ...

        app = WebhookApp(ui=ui, webhook_secret="my_secret_key")

        @app.add_webhook("/say_hello")
        async def hello(payload: WebhookPayload):
            return {"message": "hello"}

        app.run()
        ```
    """

    def __init__(
        self,
        ui: Optional[gr.Blocks] = None,
        webhook_secret: Optional[str] = None,
    ) -> None:
        self._ui = ui

        self.webhook_secret = webhook_secret or os.getenv("WEBHOOK_SECRET")
        self.registered_webhooks: Dict[str, Callable] = {}
        _warn_on_empty_secret(self.webhook_secret)

    def add_webhook(self, path: Optional[str] = None) -> Callable:
        """
        Decorator to add a webhook to the [`WebhookApp`] server.

        Args:
            path (`str`, optional):
                The URL path to register the webhook function. If not provided, the function name will be used as the
                path. In any case, all webhooks are registered under the `/webhooks` path.

        Raises:
            ValueError: If the provided path is already registered as a webhook.

        Example:
            ```python
            from huggingface_hub import WebhookApp, WebhookPayload

            app = WebhookApp()

            @app.add_webhook
            async def trigger_training(payload: WebhookPayload):
                if payload.repo.type == "dataset" and payload.event.action == "update":
                    # Trigger a training job if a dataset is updated
                    ...

            app.run()
        ```
        """
        # Usage: directly as decorator. Example: `@app.add_webhook`
        if callable(path):
            # If path is a function, it means it was used as a decorator without arguments
            return self.add_webhook()(path)

        # Usage: provide a path. Example: `@app.add_webhook(...)`
        @wraps(FastAPI.post)
        def _inner_post(*args, **kwargs):
            func = args[0]
            abs_path = f"/webhooks/{(path or func.__name__).strip('/')}"
            if abs_path in self.registered_webhooks:
                raise ValueError(f"Webhook {abs_path} already exists.")
            self.registered_webhooks[abs_path] = func

        return _inner_post

    def run(self) -> None:
        """Starts the Gradio app with the FastAPI server and registers the webhooks."""
        ui = self._ui or self._get_default_ui()

        # Start Gradio App
        #   - as non-blocking so that webhooks can be added afterwards
        #   - as shared if launch locally (to debug webhooks)
        self.fastapi_app, _, _ = ui.launch(prevent_thread_lock=True, share=_is_local)
        self.fastapi_app.middleware("http")(self._webhook_secret_middleware)

        # Register webhooks to FastAPI app
        for path, func in self.registered_webhooks.items():
            self.fastapi_app.post(path)(func)

        # Print instructions and block main thread
        url = (ui.share_url or ui.local_url).strip("/")
        message = "\nWebhooks are correctly setup and ready to use:"
        message += "\n" + "\n".join(f"  - POST {url}{webhook}" for webhook in self.registered_webhooks)
        message += "\nGo to https://huggingface.co/settings/webhooks to setup your webhooks."
        print(message)

        ui.block_thread()

    def _get_default_ui(self) -> gr.Blocks:
        """Default UI if not provided (lists webhooks and provides basic instructions)."""
        with gr.Blocks() as ui:
            gr.Markdown("# This is an app to process 🤗 Webhooks")
            gr.Markdown(
                "Webhooks are a foundation for MLOps-related features. They allow you to listen for new changes on"
                " specific repos or to all repos belonging to particular set of users/organizations (not just your"
                " repos, but any repo). Check out this [guide](https://huggingface.co/docs/hub/webhooks) to get to"
                " know more about webhooks on the Huggingface Hub."
            )
            gr.Markdown(
                f"{len(self.registered_webhooks)} webhook(s) are registered:"
                + "\n\n"
                + "\n ".join(
                    f"- [{webhook_path}]({_get_webhook_doc_url(webhook.__name__, webhook_path)})"
                    for webhook_path, webhook in self.registered_webhooks.items()
                )
            )
            gr.Markdown(
                "Go to https://huggingface.co/settings/webhooks to setup your webhooks."
                + "\nYou app is running locally. Please look at the logs to check the full URL you need to set."
                if _is_local
                else (
                    "\nThis app is running on a Space. You can find the corresponding URL in the options menu"
                    " (top-right) > 'Embed the Space'. The URL looks like 'https://{username}-{repo_name}.hf.space'."
                )
            )
        return ui

    async def _webhook_secret_middleware(
        self, request: Request, call_next: Callable[[Request], Awaitable[JSONResponse]]
    ) -> JSONResponse:
        """Middleware to check "X-Webhook-Secret" header on every webhook request."""
        if request.url.path in self.registered_webhooks:
            if self.webhook_secret is not None:
                request_secret = request.headers.get("x-webhook-secret")
                if request_secret is None:
                    return JSONResponse({"error": "x-webhook-secret header not set."}, status_code=401)
                if request_secret != self.webhook_secret:
                    return JSONResponse({"error": "Invalid webhook secret."}, status_code=403)
        return await call_next(request)


def hf_webhook(path: Optional[str] = None) -> Callable:
    """Decorator to start a [`WebhookApp`] and register the decorated function as a webhook endpoint.

    This is an helper to get started quickly. If you need more flexibility (custom landing page or webhook secret),
    please use [`WebhookApp`] directly.

    Args:
        path (`str`, optional):
            The URL path to register the webhook function. If not provided, the function name will be used as the path.
            In any case, all webhooks are registered under the `/webhooks` path.

    Example:
        ```python
        from huggingface_hub import hf_webhook, WebhookPayload

        @hf_webhook
        async def trigger_training(payload: WebhookPayload):
            if payload.repo.type == "dataset" and payload.event.action == "update":
                # Trigger a training job if a dataset is updated
                ...
        ```
    """
    if callable(path):
        # If path is a function, it means it was used as a decorator without arguments
        return hf_webhook()(path)

    @wraps(WebhookApp.add_webhook)
    def _inner(func: Callable) -> None:
        app = _get_global_app()
        app.add_webhook(path)(func)
        if len(app.registered_webhooks) == 1:
            # Register `app.run` to run at exit (only once)
            atexit.register(app.run)

    return _inner


def _get_global_app() -> WebhookApp:
    global _global_app
    if _global_app is None:
        _global_app = WebhookApp()
    return _global_app


def _warn_on_empty_secret(webhook_secret: Optional[str]) -> None:
    if webhook_secret is None:
        print("Webhook secret is not defined. This means your webhook endpoints will be open to everyone.")
        print(
            "To add a secret, set `WEBHOOK_SECRET` as environment variable or pass it at initialization: "
            "\n\t`app = GradioWebhookApp(webhook_secret='my_secret', ...)`"
        )
        print(
            "For more details about webhook secrets, please refer to"
            " https://huggingface.co/docs/hub/webhooks#webhook-secret."
        )
    else:
        print("Webhook secret is correctly defined.")


def _get_webhook_doc_url(webhook_name: str, webhook_path: str) -> str:
    """Returns the anchor to a given webhook in the docs (experimental)"""
    return "/docs#/default/" + webhook_name + webhook_path.replace("/", "_") + "_post"
