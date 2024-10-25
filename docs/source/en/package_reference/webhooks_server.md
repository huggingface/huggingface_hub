<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Webhooks Server

Webhooks are a foundation for MLOps-related features. They allow you to listen for new changes on specific repos or to
all repos belonging to particular users/organizations you're interested in following. To learn
more about webhooks on the Huggingface Hub, you can read the Webhooks [guide](https://huggingface.co/docs/hub/webhooks).

<Tip>

Check out this [guide](../guides/webhooks_server) for a step-by-step tutorial on how to setup your webhooks server and
deploy it as a Space.

</Tip>

<Tip warning={true}>

This is an experimental feature. This means that we are still working on improving the API. Breaking changes might be
introduced in the future without prior notice. Make sure to pin the version of `huggingface_hub` in your requirements.
A warning is triggered when you use an experimental feature. You can disable it by setting `HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1` as an environment variable.

</Tip>

## Server

The server is a [Gradio](https://gradio.app/) app. It has a UI to display instructions for you or your users and an API
to listen to webhooks. Implementing a webhook endpoint is as simple as decorating a function. You can then debug it
by redirecting the Webhooks to your machine (using a Gradio tunnel) before deploying it to a Space.

### WebhooksServer

[[autodoc]] huggingface_hub.WebhooksServer

### @webhook_endpoint

[[autodoc]] huggingface_hub.webhook_endpoint

## Payload

[`WebhookPayload`] is the main data structure that contains the payload from Webhooks. This is
a `pydantic` class which makes it very easy to use with FastAPI. If you pass it as a parameter to a webhook endpoint, it
will be automatically validated and parsed as a Python object.

For more information about webhooks payload, you can refer to the Webhooks Payload [guide](https://huggingface.co/docs/hub/webhooks#webhook-payloads).

[[autodoc]] huggingface_hub.WebhookPayload

### WebhookPayload

[[autodoc]] huggingface_hub.WebhookPayload

### WebhookPayloadComment

[[autodoc]] huggingface_hub.WebhookPayloadComment

### WebhookPayloadDiscussion

[[autodoc]] huggingface_hub.WebhookPayloadDiscussion

### WebhookPayloadDiscussionChanges

[[autodoc]] huggingface_hub.WebhookPayloadDiscussionChanges

### WebhookPayloadEvent

[[autodoc]] huggingface_hub.WebhookPayloadEvent

### WebhookPayloadMovedTo

[[autodoc]] huggingface_hub.WebhookPayloadMovedTo

### WebhookPayloadRepo

[[autodoc]] huggingface_hub.WebhookPayloadRepo

### WebhookPayloadUrl

[[autodoc]] huggingface_hub.WebhookPayloadUrl

### WebhookPayloadWebhook

[[autodoc]] huggingface_hub.WebhookPayloadWebhook
