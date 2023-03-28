import unittest
from unittest.mock import patch

from huggingface_hub.utils import is_gradio_available

from .testing_utils import capture_output, require_webhooks


if is_gradio_available():
    import gradio as gr
    from fastapi.testclient import TestClient

    import huggingface_hub._webhooks_server
    from huggingface_hub import WebhookPayload, WebhooksServer


# Taken from https://huggingface.co/docs/hub/webhooks#event
WEBHOOK_PAYLOAD_EXAMPLE = {
    "event": {"action": "create", "scope": "discussion"},
    "repo": {
        "type": "model",
        "name": "gpt2",
        "id": "621ffdc036468d709f17434d",
        "private": False,
        "url": {"web": "https://huggingface.co/gpt2", "api": "https://huggingface.co/api/models/gpt2"},
        "owner": {"id": "628b753283ef59b5be89e937"},
    },
    "discussion": {
        "id": "6399f58518721fdd27fc9ca9",
        "title": "Update co2 emissions",
        "url": {
            "web": "https://huggingface.co/gpt2/discussions/19",
            "api": "https://huggingface.co/api/models/gpt2/discussions/19",
        },
        "status": "open",
        "author": {"id": "61d2f90c3c2083e1c08af22d"},
        "num": 19,
        "isPullRequest": True,
        "changes": {"base": "refs/heads/main"},
    },
    "comment": {
        "id": "6399f58518721fdd27fc9caa",
        "author": {"id": "61d2f90c3c2083e1c08af22d"},
        "content": "Add co2 emissions information to the model card",
        "hidden": False,
        "url": {"web": "https://huggingface.co/gpt2/discussions/19#6399f58518721fdd27fc9caa"},
    },
    "webhook": {"id": "6390e855e30d9209411de93b", "version": 3},
}


@require_webhooks
class TestWebhooksServerDontRun(unittest.TestCase):
    def test_add_webhook_implicit_path(self):
        # Test adding a webhook
        app = WebhooksServer()

        @app.add_webhook
        async def handler():
            pass

        self.assertIn("/webhooks/handler", app.registered_webhooks)

    def test_add_webhook_explicit_path(self):
        # Test adding a webhook
        app = WebhooksServer()

        @app.add_webhook(path="/test_webhook")
        async def handler():
            pass

        self.assertIn("/webhooks/test_webhook", app.registered_webhooks)  # still registered under /webhooks

    def test_add_webhook_twice_should_fail(self):
        # Test adding a webhook
        app = WebhooksServer()

        @app.add_webhook("my_webhook")
        async def test_webhook():
            pass

        # Registering twice the same webhook should raise an error
        with self.assertRaises(ValueError):

            @app.add_webhook("my_webhook")
            async def test_webhook_2():
                pass


@require_webhooks
class TestWebhooksServerRun(unittest.TestCase):
    def setUp(self) -> None:
        with gr.Blocks() as ui:
            gr.Markdown("Hello World!")
        app = WebhooksServer(ui=ui, webhook_secret="my_webhook_secret")

        @app.add_webhook
        async def test_webhook(payload: WebhookPayload) -> None:
            return {"scope": payload.event.scope}

        self.ui = ui
        self.app = app

    def tearDown(self) -> None:
        self.ui.server.close()

    def mocked_run_app(self) -> "TestClient":
        with patch.object(self.ui, "block_thread"):
            # Run without blocking
            with patch.object(huggingface_hub._webhooks_server, "_is_local", False):
                # Run without tunnel
                self.app.run()
                return TestClient(self.app.fastapi_app)

    def test_run_print_instructions(self):
        # Test running the app
        with capture_output() as output:
            self.mocked_run_app()

        instructions = output.getvalue()
        self.assertIn("Webhooks are correctly setup and ready to use:", instructions)
        self.assertIn("- POST http://127.0.0.1:7860/webhooks/test_webhook", instructions)

    def test_run_parse_payload(self):
        client = self.mocked_run_app()
        response = client.post(
            "/webhooks/test_webhook",
            headers={"x-webhook-secret": "my_webhook_secret"},
            json=WEBHOOK_PAYLOAD_EXAMPLE,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"scope": "discussion"})

    def test_no_webhook_secret_should_fail(self):
        client = self.mocked_run_app()

        response = client.post("/webhooks/test_webhook")
        assert response.status_code == 401
