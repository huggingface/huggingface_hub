<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Webhooks Server

Les webhooks sont une fondation pour les fonctionnalités liées au MLOps. Ils vous permettent d'écouter les nouveaux changements sur des dépôts spécifiques ou sur
tous les dépôts appartenant à des utilisateurs/organisations particuliers que vous souhaitez suivre. Pour en savoir
plus sur les webhooks sur le Huggingface Hub, vous pouvez lire le [guide](https://huggingface.co/docs/hub/webhooks) Webhooks.

> [!TIP]
> Consultez ce [guide](../guides/webhooks_server) pour un tutoriel étape par étape sur comment configurer votre serveur webhooks et
> le déployer comme un Space.

> [!WARNING]
> Ceci est une fonctionnalité expérimentale. Cela signifie que nous travaillons toujours sur l'amélioration de l'API. Des changements cassants pourraient être
> introduits dans le futur sans préavis. Assurez-vous de fixer la version de `huggingface_hub` dans vos requirements.
> Un avertissement est déclenché lorsque vous utilisez une fonctionnalité expérimentale. Vous pouvez le désactiver en définissant `HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1` comme variable d'environnement.

## Serveur

Le serveur est une app [Gradio](https://gradio.app/). Il a une UI pour afficher des instructions pour vous ou vos utilisateurs et une API
pour écouter les webhooks. Implémenter un endpoint webhook est aussi simple que de décorer une fonction. Vous pouvez ensuite le déboguer
en redirigeant les Webhooks vers votre machine (en utilisant un tunnel Gradio) avant de le déployer sur un Space.

### WebhooksServer

[[autodoc]] huggingface_hub.WebhooksServer

### @webhook_endpoint

[[autodoc]] huggingface_hub.webhook_endpoint

## Payload

[`WebhookPayload`] est la structure de données principale qui contient le payload des Webhooks. C'est
une classe `pydantic` ce qui la rend très facile à utiliser avec FastAPI. Si vous la passez comme paramètre à un endpoint webhook, elle
sera automatiquement validée et analysée en tant qu'objet Python.

Pour plus d'informations sur le payload des webhooks, vous pouvez vous référer au [guide](https://huggingface.co/docs/hub/webhooks#webhook-payloads) Webhooks Payload.

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
