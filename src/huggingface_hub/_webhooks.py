import atexit
import os
from functools import wraps
from typing import Awaitable, Callable, Dict, Optional, Iterable

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


_global_app: Optional["WebhookApp"] = None
_is_local = os.getenv("SYSTEM") != "spaces"


class WebhookApp:
    """
    ```py
    from huggingface_hub import WebhookApp

    app = WebhookApp()

    @app.add_webhook("/test_webhook")
    async def hello():
        return {"in_gradio": True}

    app.run()
    ```

    ```py
    from huggingface_hub import as_webhook

    @as_webhook
    async def hello():
        return {"in_gradio": True}
    ```
    """

    def __init__(
        self,
        ui: Optional[gr.Blocks] = None,
        webhook_secret: Optional[str] = None,
    ) -> None:
        """Initialize WebhookApp.

        At this stage, it is an empty wrapper. The Gradio (and FastAPI) app will be created when `run` is called.
        """
        self._ui = ui

        self.webhook_secret = webhook_secret or os.getenv("WEBHOOK_SECRET")
        self.registered_webhooks: Dict[str, Callable] = {}
        _warn_on_empty_secret(self.webhook_secret)

    def add_webhook(self, path: Optional[str] = None) -> Callable:
        """Decorator to add a webhook to the server app."""

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
        """Set the app as "ready" and block main thread to keep it running."""
        ui = self._ui or self._get_default_ui()

        # Start Gradio App
        #   - as non-blocking so that webhooks can be added afterwards
        #   - as shared if launch locally (to debug webhooks)
        fastapi_app, _, _ = ui.launch(prevent_thread_lock=True, share=_is_local)
        fastapi_app.middleware("http")(self._webhook_secret_middleware)

        # Register webhooks to FastAPI app
        for path, func in self.registered_webhooks.items():
            fastapi_app.post(path)(func)

        # Print instructions and block main thread
        url = (ui.share_url or ui.local_url).strip("/")
        message = "\nWebhooks are correctly setup and ready to use:"
        message += "\n" + "\n".join(f"  - POST {url}{webhook}" for webhook in self.registered_webhooks)
        message += "\nGo to https://huggingface.co/settings/webhooks to setup your webhooks."
        print(message)

        ui.block_thread()

    def _get_default_ui(self) -> gr.Blocks:
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


def as_webhook(path: Optional[str] = None) -> Callable:
    """Decorator to start a webhook server app."""

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
