<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Webhooks

Les webhooks sont importants pour les fonctionnalités liées au MLOps. Ils vous permettent d'écouter les nouveaux changements sur des dépôts spécifiques. Par exemple, sur tous les dépôts appartenant à des utilisateurs/organisations particuliers. Ce guide vous expliquera d'abord comment gérer les webhooks avec du code. Ensuite, nous verrons comment utiliser `huggingface_hub` pour créer un serveur écoutant les webhooks et comment le déployer sur un Space.

Ce guide suppose que vous êtes familier avec le concept de webhooks sur le Hugging Face Hub. Pour en savoir plus sur les webhooks eux-mêmes, vous devriez d'abord lire ce [guide](https://huggingface.co/docs/hub/webhooks).

## Gérer les Webhooks

`huggingface_hub` vous permet de gérer vos webhooks avec du code. Vous pouvez lister vos webhooks existants, en créer de nouveaux, les mettre à jour, les activer/désactiver ou carrément les supprimer. Cette section vous guidera sur les procédures et fonctions API du Hugging Face Hub à utiliser.

### Créer un Webhook

Pour créer un nouveau webhook, utilisez [`create_webhook`] et spécifiez l'URL où les payloads doivent être envoyés, quels événements doivent être surveillés. Vous pouvez aussi optionnellement définir un domaine et un secret pour plus de sécurité.

```python
from huggingface_hub import create_webhook

# Exemple : Créer un webhook
webhook = create_webhook(
    url="https://webhook.site/your-custom-url",
    watched=[{"type": "user", "name": "your-username"}, {"type": "org", "name": "your-org-name"}],
    domains=["repo", "discussion"],
    secret="your-secret"
)
```

Un webhook peut également déclencher un Job qui s'exécutera sur l'infrastructure Hugging Face au lieu d'envoyer le payload à une URL. Dans ce cas, vous devez passer l'ID d'un Job.

```python
from huggingface_hub import create_webhook

# Exemple : Créer un webhook qui déclenche un Job
webhook = create_webhook(
    job_id=job_id,
    watched=[{"type": "user", "name": "your-username"}, {"type": "org", "name": "your-org-name"}],
    domains=["repo", "discussion"],
    secret="your-secret"
)
```

Le webhook déclenche le Job avec le payload du webhook dans la variable d'environnement `WEBHOOK_PAYLOAD`. Pour plus d'informations sur les Jobs Hugging Face, le hardware disponible (CPU, GPU) et les scripts UV, consultez la [documentation Jobs](./jobs).

### Lister les Webhooks

Pour voir tous les webhooks que vous avez configurés, vous pouvez les lister avec [`list_webhooks`]. Cela vous retournera leurs IDs, URLs et statuts.

```python
from huggingface_hub import list_webhooks

# Exemple : Lister tous les webhooks
webhooks = list_webhooks()
for webhook in webhooks:
    print(webhook)
```

### Mettre à jour un Webhook

Si vous devez changer la configuration d'un webhook existant, comme l'URL ou les événements qu'il surveille, vous pouvez le mettre à jour en utilisant [`update_webhook`].

```python
from huggingface_hub import update_webhook

# Exemple : Mettre à jour un webhook
updated_webhook = update_webhook(
    webhook_id="your-webhook-id",
    url="https://new.webhook.site/url",
    watched=[{"type": "user", "name": "new-username"}],
    domains=["repo"]
)
```

### Activer et Désactiver les Webhooks

Vous pourriez vouloir désactiver temporairement un webhook sans le supprimer. Cela peut être fait en utilisant [`disable_webhook`].Le webhook peut être réactivé plus tard avec [`enable_webhook`].

```python
from huggingface_hub import enable_webhook, disable_webhook

# Exemple : Activer un webhook
enabled_webhook = enable_webhook("your-webhook-id")
print("Enabled:", enabled_webhook)

# Exemple : Désactiver un webhook
disabled_webhook = disable_webhook("your-webhook-id")
print("Disabled:", disabled_webhook)
```

### Supprimer un Webhook

Lorsqu'un webhook n'est plus nécessaire, il peut être définitivement supprimé en utilisant [`delete_webhook`].

```python
from huggingface_hub import delete_webhook

# Exemple : Supprimer un webhook
delete_webhook("your-webhook-id")
```

## Serveur Webhooks

La classe de base que nous utiliserons dans cette section est [`WebhooksServer`]. C'est une classe pour configurer facilement un serveur qui peut recevoir des webhooks depuis le Hugging Face Hub. Le serveur est basé sur [Gradio](https://gradio.app/).

> [!TIP]
> Pour voir un exemple en cours d'exécution d'un serveur webhook, consultez le [Spaces CI Bot](https://huggingface.co/spaces/spaces-ci-bot/webhook). C'est un Space qui lance des environnements éphémères lorsqu'une PR est ouverte sur un Space.

> [!WARNING]
> Ceci est une [fonctionnalité expérimentale](../package_reference/environment_variables#hfhubdisableexperimentalwarning). Cela signifie que nous travaillons toujours sur l'amélioration de l'API. Des changements cassants pourraient être introduits dans le futur sans préavis. Assurez-vous de fixer la version de `huggingface_hub` dans vos requirements.

### Créer un endpoint

Voyons un premier exemple pour expliquer les concepts principaux :

```python
# app.py
from huggingface_hub import webhook_endpoint, WebhookPayload

@webhook_endpoint
async def trigger_training(payload: WebhookPayload) -> None:
    if payload.repo.type == "dataset" and payload.event.action == "update":
        # Déclencher un job d'entraînement si un dataset est mis à jour
        ...
```

Enregistrez cet extrait dans un fichier appelé `'app.py'` et exécutez-le avec `'python app.py'`. Vous devriez voir un message comme celui-ci :

```text
Webhook secret is not defined. This means your webhook endpoints will be open to everyone.
To add a secret, set `WEBHOOK_SECRET` as environment variable or pass it at initialization:
        `app = WebhooksServer(webhook_secret='my_secret', ...)`
For more details about webhook secrets, please refer to https://huggingface.co/docs/hub/webhooks#webhook-secret.
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://1fadb0f52d8bf825fc.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades (NEW!), check out Spaces: https://huggingface.co/spaces

Webhooks are correctly setup and ready to use:
  - POST https://1fadb0f52d8bf825fc.gradio.live/webhooks/trigger_training
Go to https://huggingface.co/settings/webhooks to setup your webhooks.
```

Bon travail ! Vous venez de lancer un serveur webhook ! Décomposons ce qui s'est passé exactement :

1. En décorant une fonction avec [`webhook_endpoint`], un objet [`WebhooksServer`] a été créé en arrière-plan. Comme vous pouvez le voir, ce serveur est une app Gradio fonctionnant sur http://127.0.0.1:7860. Si vous ouvrez cette URL dans votre navigateur, vous verrez une page d'accueil avec des instructions sur les webhooks enregistrés.
2. Une app Gradio est un serveur FastAPI en réalité. Une nouvelle route POST `/webhooks/trigger_training` a été ajoutée à celui-ci. C'est la route qui écoutera les webhooks et exécutera la fonction `trigger_training` lorsqu'elle sera déclenchée. FastAPI analysera automatiquement le payload et le passera à la fonction en tant qu'objet [`WebhookPayload`]. C'est un objet `pydantic` qui contient toutes les informations sur l'événement qui a déclenché le webhook.
3. L'app Gradio a également ouvert un tunnel pour recevoir des requêtes depuis Internet. C'est la partie intéressante : vous pouvez configurer un Webhook sur https://huggingface.co/settings/webhooks pointant vers votre machine locale. Cela est utile pour déboguer votre serveur webhook et itérer rapidement avant de le déployer sur un Space.
4. Enfin, les logs vous disent également que votre serveur n'est actuellement pas sécurisé par un secret. Ce n'est pas problématique pour une configuration locale mais c'est à garder à l'esprit pour plus tard.

> [!WARNING]
> Par défaut, le serveur est démarré à la fin de votre script. Si vous l'exécutez dans un notebook, vous pouvez démarrer le serveur manuellement en appelant `decorated_function.run()`.

### Configurer un Webhook

Maintenant que vous avez un serveur webhook en cours d'exécution, vous voulez configurer un Webhook pour commencer à recevoir des messages. Allez sur https://huggingface.co/settings/webhooks, cliquez sur "Add a new webhook" et configurez votre Webhook. Définissez les dépôts cibles que vous voulez surveiller et l'URL du Webhook, ici `https://1fadb0f52d8bf825fc.gradio.live/webhooks/trigger_training`.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/configure_webhook.png"/>
</div>

Et voilà ! Vous pouvez maintenant déclencher ce webhook en mettant à jour le dépôt cible (par exemple push un commit). Consultez l'onglet Activité de votre Webhook pour voir les événements qui ont été déclenchés. Si vous modifiez votre code et redémarrez le serveur, votre URL publique pourrait changer. Assurez-vous de mettre à jour la configuration du webhook sur le Hub si nécessaire.

### Déployer sur un Space

Maintenant que vous avez un serveur webhook fonctionnel, l'objectif est de le déployer sur un Space. Allez sur https://huggingface.co/new-space pour créer un Space. Donnez-lui un nom, sélectionnez le SDK Gradio et cliquez sur "Create Space". Uploadez votre code vers le Space dans un fichier appelé `app.py`. Votre Space démarrera automatiquement ! Pour plus de détails sur les Spaces, veuillez vous référer à ce [guide](https://huggingface.co/docs/hub/spaces-overview).

Votre serveur webhook fonctionne maintenant sur un Space public. Dans la plupart des cas, vous voudrez le sécuriser avec un secret. Allez dans les paramètres de votre Space > Section "Repository secrets" > "Add a secret". Définissez la variable d'environnement `WEBHOOK_SECRET` à la valeur de votre choix. Retournez aux [paramètres Webhooks](https://huggingface.co/settings/webhooks) et définissez le secret dans la configuration du webhook. Maintenant, seules les requêtes avec le bon secret seront acceptées par votre serveur.

Et c'est tout ! Votre Space est maintenant prêt à recevoir des webhooks depuis le Hub. Veuillez garder à l'esprit que si vous exécutez le Space sur un hardware gratuit 'cpu-basic', il sera arrêté après 48 heures d'inactivité. Si vous avez besoin d'un Space permanent, vous devriez envisager de passer à un [hardware amélioré](https://huggingface.co/docs/hub/spaces-gpus#hardware-specs).

### Utilisation avancée

Le guide ci-dessus a expliqué le moyen le plus rapide de configurer un [`WebhooksServer`]. Dans cette section, nous verrons comment le personnaliser davantage.

#### Plusieurs endpoints

Vous pouvez enregistrer plusieurs endpoints sur le même serveur. Par exemple, vous pourriez vouloir avoir un endpoint pour déclencher un job d'entraînement et un autre pour déclencher une évaluation de modèle. Vous pouvez le faire en ajoutant plusieurs décorateurs `@webhook_endpoint` :

```python
# app.py
from huggingface_hub import webhook_endpoint, WebhookPayload

@webhook_endpoint
async def trigger_training(payload: WebhookPayload) -> None:
    if payload.repo.type == "dataset" and payload.event.action == "update":
        # Déclencher un job d'entraînement si un dataset est mis à jour
        ...

@webhook_endpoint
async def trigger_evaluation(payload: WebhookPayload) -> None:
    if payload.repo.type == "model" and payload.event.action == "update":
        # Déclencher un job d'évaluation si un modèle est mis à jour
        ...
```

Ce qui créera deux endpoints :

```text
(...)
Webhooks are correctly setup and ready to use:
  - POST https://1fadb0f52d8bf825fc.gradio.live/webhooks/trigger_training
  - POST https://1fadb0f52d8bf825fc.gradio.live/webhooks/trigger_evaluation
```

#### Serveur personnalisé

Pour obtenir plus de flexibilité, vous pouvez également créer un objet [`WebhooksServer`] directement. Ceci est utile si vous voulez personnaliser la page d'accueil de votre serveur. Vous pouvez le faire en passant une [UI Gradio](https://gradio.app/docs/#blocks) qui écrasera celle par défaut. Lors de la création d'un [`WebhooksServer`], vous pouvez enregistrer de nouveaux webhooks en utilisant le décorateur [`~WebhooksServer.add_webhook`].

Voici un exemple complet :

```python
import gradio as gr
from fastapi import Request
from huggingface_hub import WebhooksServer, WebhookPayload

# 1. Définir l'UI
with gr.Blocks() as ui:
    ...

# 2. Créer WebhooksServer avec une UI personnalisée et un secret
app = WebhooksServer(ui=ui, webhook_secret="my_secret_key")

# 3. Enregistrer le webhook avec un nom explicite
@app.add_webhook("/say_hello")
async def hello(payload: WebhookPayload):
    return {"message": "hello"}

# 4. Enregistrer le webhook avec un nom implicite
@app.add_webhook
async def goodbye(payload: WebhookPayload):
    return {"message": "goodbye"}

# 5. Démarrer le serveur (optionnel)
app.run()
```

1. Nous définissons une UI personnalisée en utilisant les blocs Gradio. Cette UI sera affichée sur la page d'accueil du serveur.
2. Nous créons un objet [`WebhooksServer`] avec une UI personnalisée et un secret. Le secret est optionnel et peut être défini avec la variable d'environnement `WEBHOOK_SECRET`.
3. Nous enregistrons un webhook avec un nom explicite. Cela créera un endpoint à `/webhooks/say_hello`.
4. Nous enregistrons un webhook avec un nom implicite. Cela créera un endpoint à `/webhooks/goodbye`.
5. Nous démarrons le serveur. C'est optionnel car votre serveur sera automatiquement démarré à la fin du script.
