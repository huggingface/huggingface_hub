<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Serveurs Webhooks

Les webhooks sont un pilier des fonctionnalités MLOps. Ils vous permettent de suivre tous les nouveaux
changements sur des dépôts spécifiques ou sur tous les dépôts appartenants à des utilisateurs/organisations que
vous voulez suivre. Ce guide vous expliquera comment utiliser `hugginface_hub` pour créer un serveur écoutant des
webhooks et le déployer sur un espacce. Il postule que vous êtes familier avec le concept de webhooks sur le Hub Hugging Face.
Pour en apprendre plus sur les webhooks, vous pouvez consulter le
[guide](https://huggingface.co/docs/hub/webhooks) d'abord. 

La classe de base que nous utiliserons dans ce guide est [`WebhooksServer`]. C'est une classe qui permet de configurer
facilement un serveur qui peut recevoir des webhooks du Hub Huggingface. Le serveur est basé sur une application
[Gradio](https://gradio.app/). Il a une interface pour afficher des instruction pour vous ou vos utilisateurs et une API
pour écouter les webhooks.

<Tip>

Pour voir un exemple fonctionnel de serveur webhook, consultez le [space bot CI](https://huggingface.co/spaces/spaces-ci-bot/webhook).
C'est un space qui lance des environnements éphémères lorsqu'une pull request est ouverte sur un space.

</Tip>

<Tip warning={true}>

C'est une [fonctionnalité expérimentale](../package_reference/environment_variables#hfhubdisableexperimentalwarning),
ce qui signifie que nous travaillons toujours sur l'amélioration de l'API. De nouveaux changement pourront être introduit
dans les future sans avertissement préalable. Assurez vous d'épingler la version d'`hugginface_hub` dans vos requirements.

</Tip>


## Créer un endpoint

Implémenter un endpoint webhook est aussi facile que d'ajouter des décorateurs à une fonction. Voyons un premier
exemple afin de clarifier les concepts principaux:

```python
# app.py
from huggingface_hub import webhook_endpoint, WebhookPayload

@webhook_endpoint
async def trigger_training(payload: WebhookPayload) -> None:
    if payload.repo.type == "dataset" and payload.event.action == "update":
        # Lance une tâche d'entrainement si votre dataset est mis à jour
        ...
```

Enregistrez ce snippet de code dans un fichier appelé `'app.py'` et lancez le avec `'python app.py'`. Vous devriez
voir un message de ce type:

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

Bien joué! Vous venez de créer un serveur webhook! Décomposons ce qui vient de se passer exactement:

1. En décorant une fonction avec [`webhook_endpoint`], un objet [`WebhooksServer`] a été créé en arrière plan.
Comme vous pouvez le voir, ce serveur est une application Gradio qui tourne sur http://127.0.0.1:7860. Si vous ouvrez
cet URL dans votre navigateur, vous verez une page d'accueil avec des informations sur les webhooks en question.
2. En arrière plan, une application Gradio n'est rien de plus qu'un serveur FastAPI . Une nouvelle route POST `/webhooks/trigger_training`
y a été ajoutée. C'est cette route qui écoutera les webhooks et lancera la fonction `trigger_training` lors de son activation.
FastAPI parsera automatiquement le paquet et le passera à la fonction en tant qu'objet [`WebhookPayload`]. C'est un objet
`pydantic` qui contient toutes les informations sur l'événement qui a activé le webhook.
3. L'application Gradio a aussi ouvert un tunnel pour recevoir des requêtes d'internet. C'est la partie intéressante:
vous pouvez configurer un webhooj sur https://huggingface.co/settings/webhooks qui pointe vers votre machine. C'est utile
pour debugger votre serveur webhook et rapidement itérer avant de le déployer dans un space.
4. Enfin, les logs vous disent aussi que votre serveur n'est pas sécurisé par un secret. Ce n'est pas problématique pour 
le debugging en local mais c'est à garder dans un coin de la tête pour plus tard.

<Tip warning={true}>

Par défaut, le serveur est lancé à la fin de votre script. Si vous l'utilisez dans un notebook, vous pouvez lancer serveur
manuellement en appelant `decorated_function.run()`. Vu qu'un unique serveur est utilisé, vous aurez uniquement besoin de
lancer le serveur une fois même si vous avez plusieurs endpoints.

</Tip>


## Configurer un webhook

Maintenant que vous avez un serveur webhook qui tourne, vous aurez surement besoin de configure un webhook
pour commencer à recevoir des messages. Allez sur https://huggingface.co/settings/webhooks, cliquez sur
"Ajouter un nouveau webhook" et configurez votre webhook. Définissez les dépôts cibles que vous voulez
surveiller et l'URL du webhook, ici `https://1fadb0f52d8bf825fc.gradio.live/webhooks/trigger_training`.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/configure_webhook.png"/>
</div>

Et voilà! Vous pouvez maintenant activer cce webhook en mettant à jour le dépôt cible (i.e. push un commit). Consultez
la table d'activité de votre webhook pour voir les événements passés. Maintenant que vous avez un setup qui fonctionne,
vous pouvez le tester et itérer rapidement. Si vous modifiez votre code et relancez le serveur, votre URL public pourrait
changer. Assurez vous de mettre à jour la configuration du webhook dans le Hub si besoin.

## Déployer vers un space

Maintenant que vous avez un serveur webhook fonctionnel, le but est de le déployer sur un space. Allez sur
https://huggingface.co/new-space pour créer un space. Donnez lui un nom, sélectionnez le SDK Gradio et cliquer sur
"Créer un space" (ou "Create Space" en anglais). Uploadez votre code dans le space dans un fichier appelé `app.py`.
Votre space sera lancé automatiquement! Pour plus de détails sur les spaces, consultez ce [guide](https://huggingface.co/docs/hub/spaces-overview).

Votre serveur webhook tourne maintenant sur un space public. Dans la plupart de cas, vous aurez besoin de le sécuriser
avec un secret. Allez dans les paramètres de votre space > Section "Repository secrets" > "Add a secret". Définissez
la variable d'environnement `WEBHOOK_SECRET` en choisissant la valeur que vous voulez. Retournez dans les 
[réglages webhooks](https://huggingface.co/settings/webhooks) et définissez le secret dans la configuration du webhook.
Maintenant, seules les requêtes avec le bon secret seront acceptées par votre serveur.

Et c'est tout! Votre space est maintenant prêt à recevoir des webhooks du Hub. Gardez à l'esprit que si vous faites
tourner le space sur le hardware gratuit 'cpu-basic', il sera éteint après 48 heures d'inactivité. Si vous avez besoin d'un
space permanent, vous devriez peut-être considérer l'amélioration vers un [hardware amélioré](https://huggingface.co/docs/hub/spaces-gpus#hardware-specs).

## Utilisation avancée

Le guide ci dessus expliquait la manière la plus rapide d'initialiser un [`WebhookServer`]. Dans cette section, nous verrons
comment le personnaliser plus en détails.

### Endpoints multilpes

Vous pouvez avoir plusieurs endpoints sur le même serveur. Par exemple, vous aurez peut-être envie d'avoir un endpoint
qui lance un entrainement de modèle et un autre qui lance une évaluation de modèle. Vous pouvez faire ça en ajoutant
plusieurs décorateurs `@webhook_endpoint`:

```python
# app.py
from huggingface_hub import webhook_endpoint, WebhookPayload

@webhook_endpoint
async def trigger_training(payload: WebhookPayload) -> None:
    if payload.repo.type == "dataset" and payload.event.action == "update":
        # Lance une tâche d'entrainement si votre dataset est mis à jour
        ...

@webhook_endpoint
async def trigger_evaluation(payload: WebhookPayload) -> None:
    if payload.repo.type == "model" and payload.event.action == "update":
        # Lance un tâche d'évaluation si un modèle est mis à jour
        ...
```

Ce qui créera deux endpoints:

```text
(...)
Webhooks are correctly setup and ready to use:
  - POST https://1fadb0f52d8bf825fc.gradio.live/webhooks/trigger_training
  - POST https://1fadb0f52d8bf825fc.gradio.live/webhooks/trigger_evaluation
```

### Serveur personnalisé

Pour plus de flexibilité, vous pouvez aussi créer un objet [`WebhooksServer`] directement. C'est utile si vous
voulez customiser la page d'acceuil de votre serveur. Vous pouvez le faire en passant une [UI Gradio](https://gradio.app/docs/#blocks)
qui va overwrite linterface par défaut. Par exemple, vous pouvez ajouter des instructions pour vos utilisateurs
ou ajouter un formulaire pour avtiver les webhooks manuellement. Lors de la création d'un [`WebhooksServer`], vous
pouvez créer de nouveaux webhooks en utilisant le décorateur [`~WebhooksServer.add_webhook`].

Here is a complete example:

```python
import gradio as gr
from fastapi import Request
from huggingface_hub import WebhooksServer, WebhookPayload

# 1. Déifnition de l'interface
with gr.Blocks() as ui:
    ...

# 2. Création d'un WebhooksServer avec une interface personnalisée et un secret
app = WebhooksServer(ui=ui, webhook_secret="my_secret_key")

# 3. Ajout d'un webhook avec un nom explicite
@app.add_webhook("/say_hello")
async def hello(payload: WebhookPayload):
    return {"message": "hello"}

# 4. Ajout d'un webhook avec un nom implicite
@app.add_webhook
async def goodbye(payload: WebhookPayload):
    return {"message": "goodbye"}

# 5. Lancement du serveur (optionnel)
app.run()
```

1. Nous définissons une interface personnalisée en utilisant des block Gradio. Cette interface sera affichée
sur la page d'accueil du serveur.
2. Nous créons un objet [`WebhooksServer`] avec une interface personnalisée et un secret. Le secret est optionnel et
peut être définit avec la variable d'environnement `WEBHOOK_SECRET`.
3. Nous créons un webhook avec un nom explicite. Ceci créera un endpoint à `/webhooks/say_hello`.
4. Nous créons un webhook avec un nom implicite. Ceci créera un endpoint à `/webhoojs/goodbye`.
5. Nous lançons le serveur. C'est optionnel car votre serveur sera automatiquement lancé à la fin du script.