<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Créer et partager des Model Cards

La bibliothèque `huggingface_hub` fournit une interface Python pour créer, partager et mettre à jour des Model Cards. Consultez [la page de documentation dédiée](https://huggingface.co/docs/hub/models-cards) pour une vue plus approfondie de ce que sont les Model Cards sur le Hub, et comment elles fonctionnent en interne.

## Charger une Model Card depuis le Hub

Pour charger une carte existante depuis le Hub, vous pouvez utiliser la fonction [`ModelCard.load`]. Ici, nous allons charger la carte depuis [`nateraw/vit-base-beans`](https://huggingface.co/nateraw/vit-base-beans).

```python
from huggingface_hub import ModelCard

card = ModelCard.load('nateraw/vit-base-beans')
```

Cette carte a quelques attributs utiles que vous pourriez exploiter :
  - `card.data` : Retourne une instance [`ModelCardData`] avec les métadonnées de la model card. Appelez `.to_dict()` sur cette instance pour obtenir la représentation sous forme de dictionnaire.
  - `card.text` : Retourne le texte de la carte, *excluant l'en-tête de métadonnées*.
  - `card.content` : Retourne le contenu textuel de la carte, *incluant l'en-tête de métadonnées*.

## Créer des Model Cards

### Depuis le texte

Pour initialiser une Model Card depuis du texte, passez simplement le contenu textuel de la carte au `ModelCard` lors de l'initialisation.

```python
content = """
---
language: en
license: mit
---

# My Model Card
"""

card = ModelCard(content)
card.data.to_dict() == {'language': 'en', 'license': 'mit'}  # True
```

Une autre façon que vous pourriez vouloir faire cela est avec des f-strings. Dans l'exemple suivant, nous :

- Utilisons [`ModelCardData.to_yaml`] pour convertir les métadonnées que nous avons définies en YAML afin de pouvoir les utiliser pour insérer le bloc YAML dans la model card.
- Montrons comment vous pourriez utiliser une variable de template via les f-strings Python.

```python
card_data = ModelCardData(language='en', license='mit', library='timm')

example_template_var = 'nateraw'
content = f"""
---
{ card_data.to_yaml() }
---

# My Model Card

This model created by [@{example_template_var}](https://github.com/{example_template_var})
"""

card = ModelCard(content)
print(card)
```

L'exemple ci-dessus nous laisserait avec une carte qui ressemble à ceci :

```
---
language: en
license: mit
library: timm
---

# My Model Card

This model created by [@nateraw](https://github.com/nateraw)
```

### Depuis un template Jinja

Si vous avez `Jinja2` installé, vous pouvez créer des Model Cards depuis un fichier template jinja. Voyons un exemple basique :

```python
from pathlib import Path

from huggingface_hub import ModelCard, ModelCardData

# Définir votre template jinja
template_text = """
---
{{ card_data }}
---

# Model Card for MyCoolModel

This model does this and that.

This model was created by [@{{ author }}](https://hf.co/{{author}}).
""".strip()

# Écrire le template dans un fichier
Path('custom_template.md').write_text(template_text)

# Définir les métadonnées de la carte
card_data = ModelCardData(language='en', license='mit', library_name='keras')

# Créer la carte depuis le template, en lui passant toutes les variables de template jinja que vous souhaitez.
# Dans notre cas, nous passerons author
card = ModelCard.from_template(card_data, template_path='custom_template.md', author='nateraw')
card.save('my_model_card_1.md')
print(card)
```

Le markdown de la carte résultante ressemble à ceci :

```
---
language: en
license: mit
library_name: keras
---

# Model Card for MyCoolModel

This model does this and that.

This model was created by [@nateraw](https://hf.co/nateraw).
```

Si vous mettez à jour card.data, cela se reflétera dans la carte elle-même.

```
card.data.library_name = 'timm'
card.data.language = 'fr'
card.data.license = 'apache-2.0'
print(card)
```

Maintenant, comme vous pouvez le voir, l'en-tête de métadonnées a été mis à jour :

```
---
language: fr
license: apache-2.0
library_name: timm
---

# Model Card for MyCoolModel

This model does this and that.

This model was created by [@nateraw](https://hf.co/nateraw).
```

Lorsque vous mettez à jour les données de la carte, vous pouvez valider que la carte est toujours valide par rapport au Hub en appelant [`ModelCard.validate`]. Cela garantit que la carte passe toutes les règles de validation configurées sur le Hugging Face Hub.

### Depuis le template par défaut

Au lieu d'utiliser votre propre template, vous pouvez également utiliser le [template par défaut](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md), qui est une model card complète avec de nombreuses sections que vous pourriez remplir. En pratique, il utilise [Jinja2](https://jinja.palletsprojects.com/en/3.1.x/) pour remplir le template.

> [!TIP]
> Notez que vous devrez avoir Jinja2 installé pour utiliser `from_template`. Vous pouvez le faire avec `pip install Jinja2`.

```python
card_data = ModelCardData(language='en', license='mit', library_name='keras')
card = ModelCard.from_template(
    card_data,
    model_id='my-cool-model',
    model_description="this model does this and that",
    developers="Nate Raw",
    repo="https://github.com/huggingface/huggingface_hub",
)
card.save('my_model_card_2.md')
print(card)
```

## Partager des Model Cards

Si vous êtes authentifié avec le Hugging Face Hub (soit en utilisant `hf auth login` soit [`login`]), vous pouvez pousser des cartes vers le Hub en appelant simplement [`ModelCard.push_to_hub`]. Voyons comment faire cela :
²
D'abord, nous allons créer un nouveau dépôt appelé 'hf-hub-modelcards-pr-test' sous le namespace de l'utilisateur authentifié :

```python
from huggingface_hub import whoami, create_repo

user = whoami()['name']
repo_id = f'{user}/hf-hub-modelcards-pr-test'
url = create_repo(repo_id, exist_ok=True)
```

Ensuite, nous allons créer une carte depuis le template par défaut (identique à celle définie dans la section ci-dessus) :

```python
card_data = ModelCardData(language='en', license='mit', library_name='keras')
card = ModelCard.from_template(
    card_data,
    model_id='my-cool-model',
    model_description="this model does this and that",
    developers="Nate Raw",
    repo="https://github.com/huggingface/huggingface_hub",
)
```

Enfin, nous allons pousser cela vers le hub

```python
card.push_to_hub(repo_id)
```

Vous pouvez consulter la carte résultante [ici](https://huggingface.co/nateraw/hf-hub-modelcards-pr-test/blob/main/README.md).

Si vous voulez plutôt pousser une carte comme une pull request, vous pouvez simplement dire `create_pr=True` lors de l'appel à `push_to_hub` :

```python
card.push_to_hub(repo_id, create_pr=True)
```

Une PR résultante créée depuis cette commande peut être vue [ici](https://huggingface.co/nateraw/hf-hub-modelcards-pr-test/discussions/3).

## Mettre à jour les métadonnées

Dans cette section, nous allons voir quelles sont les métadonnées dans les cartes de dépôt et comment les mettre à jour.

`metadata` fait référence à un contexte de hash map (ou clé-valeur) qui fournit des informations de haut niveau sur un modèle, un dataset ou un Space. Ces informations peuvent inclure des détails tels que le `pipeline type` du modèle, `model_id` ou `model_description`. Pour plus de détails, vous pouvez consulter ces guides : [Model Card](https://huggingface.co/docs/hub/model-cards#model-card-metadata), [Dataset Card](https://huggingface.co/docs/hub/datasets-cards#dataset-card-metadata) et [Spaces Settings](https://huggingface.co/docs/hub/spaces-settings#spaces-settings). Voyons maintenant quelques exemples sur comment mettre à jour ces métadonnées.

Commençons par un premier exemple :

```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("username/my-cool-model", {"pipeline_tag": "image-classification"})
```

Avec ces deux lignes de code, vous mettrez à jour les métadonnées pour définir un nouveau `pipeline_tag`.

Par défaut, vous ne pouvez pas mettre à jour une clé qui existe déjà sur la carte. Si vous voulez le faire, vous devez passer `overwrite=True` explicitement :

```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("username/my-cool-model", {"pipeline_tag": "text-generation"}, overwrite=True)
```

Il arrive souvent que vous souhaitiez suggérer des changements à un dépôt sur lequel vous n'avez pas de permission d'écriture. Vous pouvez le faire en créant une PR sur ce dépôt qui permettra aux propriétaires de revoir et merger vos suggestions.

```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("someone/model", {"pipeline_tag": "text-classification"}, create_pr=True)
```

## Inclure des résultats d'évaluation

Pour inclure des résultats d'évaluation dans les métadonnées `model-index`, vous pouvez passer un [`EvalResult`] ou une liste de `EvalResult` avec vos résultats d'évaluation associés. En pratique, cela créera le `model-index` lorsque vous appelez `card.data.to_dict()`. Pour plus d'informations sur comment cela fonctionne, vous pouvez consulter [cette section de la documentation Hub](https://huggingface.co/docs/hub/models-cards#evaluation-results).

> [!TIP]
> Notez que l'utilisation de cette fonction nécessite que vous incluiez l'attribut `model_name` dans [`ModelCardData`].

```python
card_data = ModelCardData(
    language='en',
    license='mit',
    model_name='my-cool-model',
    eval_results = EvalResult(
        task_type='image-classification',
        dataset_type='beans',
        dataset_name='Beans',
        metric_type='accuracy',
        metric_value=0.7
    )
)

card = ModelCard.from_template(card_data)
print(card.data)
```

Le `card.data` résultant devrait ressembler à ceci :

```
language: en
license: mit
model-index:
- name: my-cool-model
  results:
  - task:
      type: image-classification
    dataset:
      name: Beans
      type: beans
    metrics:
    - type: accuracy
      value: 0.7
```

Si vous avez plus d'un résultat d'évaluation que vous aimeriez partager, passez simplement une liste de `EvalResult` :

```python
card_data = ModelCardData(
    language='en',
    license='mit',
    model_name='my-cool-model',
    eval_results = [
        EvalResult(
            task_type='image-classification',
            dataset_type='beans',
            dataset_name='Beans',
            metric_type='accuracy',
            metric_value=0.7
        ),
        EvalResult(
            task_type='image-classification',
            dataset_type='beans',
            dataset_name='Beans',
            metric_type='f1',
            metric_value=0.65
        )
    ]
)
card = ModelCard.from_template(card_data)
card.data
```

Ce qui devrait vous laisser avec le `card.data` suivant :

```
language: en
license: mit
model-index:
- name: my-cool-model
  results:
  - task:
      type: image-classification
    dataset:
      name: Beans
      type: beans
    metrics:
    - type: accuracy
      value: 0.7
    - type: f1
      value: 0.65
```
