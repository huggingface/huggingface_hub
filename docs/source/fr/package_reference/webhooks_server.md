<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Serveurs WebHooks

Les webhooks sont un pilier des fonctionnalités MLOps. Ils vous permettent de suivre tous les nouveaux
changements sur des dépôts spécifiques ou sur tous les dépôts appartenants à des utilisateurs/organisations que
vous voulez suivre. Pour en apprendre plus sur les webhooks dans le Hub Huggingface, vous pouvez consulter
le [guide](https://huggingface.co/docs/hub/webhooks) consacré aux webhooks. 

<Tip>

Consultez ce [guide](../guides/webhooks_server) pour un tutoriel pas à pas sur comment mettre en place votre serveur
webhooks et le déployer en tant que space.

</Tip>

<Tip warning={true}>

Ceci est une fonctionnalité expérimentale, ce qui signifie que nous travaillons toujours sur l'amélioration de l'API.
De gros changements pourraient être introduit dans le futur sans avertissement préalable. Faites en sorte
d'épingler la version d'`huggingface_hub` dans le requirements. Un avertissement est affiché lorsque vous utilisez
des fonctionnalités expérimentales. Vous pouvez le supprimer en définissant la variable d'environnement
`HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1`.

</Tip>

## Serveur

Le serveur est une application [Gradio](https://gradio.app/). Il possède une interface pour afficher des instructions pour vous
ou vos utilisateurs et une API pour écouter les webhooks. Implémenter un endpoint de webhook est aussi simple que d'ajouter
un décorateur à une fonction. Vous pouvez ensuite le debugger en redirigeant le webhook vers votre machine (en utilisant
un tunnel Gradio) avant de le déployer sur un space.

### WebhooksServer

[[autodoc]] huggingface_hub.WebhooksServer

### @webhook_endpoint

[[autodoc]] huggingface_hub.webhook_endpoint

## Payload

[`WebhookPayload`] est la structure de donnée principale qui contient le payload de webhooks.
C'est une classe `pydantic` ce qui la rend très facile à utiliser avec FastAPI. Si vous la
passez en tant que paramètre d'un endpoint webhook, il sera automatiquement validé et parsé en tant qu'objet Python.

Pour plus d'informations sur les payload webhooks, vous pouvez vous référer au [guide](https://huggingface.co/docs/hub/webhooks#webhook-payloads).
sur les payloads webhooks

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
