<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Créer en partager des cartes de modèle

La librairie `huggingface_hub` fournit une interface Python pour créer, partager et mettre à jour
des cartes de modèle. Consultez [page de documentation dédiée](https://huggingface.co/docs/hub/models-cards)
pour une vue en profondeur de ce que les cartes de modèle sont et comment elles fonctionnent
en arrière-plan.

<Tip>

[`Nouveaux : Testez notre application de création de carte de modèle`](https://huggingface.co/spaces/huggingface/Model_Cards_Writing_Tool)

</Tip>

## Chargez une carte de modèle depuis le Hub

Pour charger une carte existante depuis le Hub, vous pouvez utiliser la fonction [`ModelCard.load`]. Ici, nous chargeront la carte depuis [`nateraw/vit-base-beans`](https://huggingface.co/nateraw/vit-base-beans).

```python
from huggingface_hub import ModelCard

card = ModelCard.load('nateraw/vit-base-beans')
```

Cette carte a des attributs très utiles que vous aurez peut-être envie d'utiliser:
  - `card.data`: Retourne une instance [`ModelCardData`] avec les métadonnées de la carte de modèle. Les appels à `.to_dict()` sur cette instance pour obtenir les représentations en tant que dictionnaire.
  - `card.text`: Retourne le texte de la carte *sans le header des métadonnées*.
  - `card.content`: Retourne le contenu textuel de la carte, *dont le header des métadonnées*.

## Créez des cartes de modèle

### Depuis le texte

Pour initialiser une carte de modèle depuis le texte, passez simplement en tant qu'argument le
prochain contenu de la carte au `ModelCard` à l'initialisation. 

```python
content = """
---
language: en
license: mit
---

# Ma carte de modèle
"""

card = ModelCard(content)
card.data.to_dict() == {'language': 'en', 'license': 'mit'}  # True
```

Une autre manière que vous aurez peut-être besoin d'utiliser est celle utilisant les
f-strings. Dans l'exemple suivant nous:

- Utiliseront [`ModelCardData.to_yaml`] pour convertir la métadonnée que nous avons définie en YAML afin de pouvoir l'utiliser pour
  insérer le block YAML dans la carte de modèle.
- Montreront comment vous pourriez utiliser une variable dans un template via les f-strings Python.

```python
card_data = ModelCardData(language='en', license='mit', library='timm')

example_template_var = 'nateraw'
content = f"""
---
{ card_data.to_yaml() }
---

# Ma carte de modèle

Ce modèle créé par [@{example_template_var}](https://github.com/{example_template_var})
"""

card = ModelCard(content)
print(card)
```

L'exemple ci-dessus nous laisserait avec une carte qui ressemble à ça:

```
---
language: en
license: mit
library: timm
---

# Ma carte de modèle

Ce modèle a été créé par [@nateraw](https://github.com/nateraw)
```

### Depuis un template Jinja

Si `Jinja2` est installé, vous pouvez créer une carte de modèle depuis un template jinja. Consultons un exemple
basique:

```python
from pathlib import Path

from huggingface_hub import ModelCard, ModelCardData

# Définissez votre template jinja
template_text = """
---
{{ card_data }}
---

# Carte de modèle de MyCoolModel

Ce modèle fait ceci, il peut aussi faire cela...

Ce modèle a été créé par [@{{ author }}](https://hf.co/{{author}}).
""".strip()

# Écrivez le template vers un fichier
Path('custom_template.md').write_text(template_text)

# Définissez la métadonnée de la carte
card_data = ModelCardData(language='en', license='mit', library_name='keras')

# Créez une carte depuis le template vous pouvez passer n'importe quelle variable de template jinja que vous voulez.
# Dans notre cas, nous passeront author
card = ModelCard.from_template(card_data, template_path='custom_template.md', author='nateraw')
card.save('my_model_card_1.md')
print(card)
```

Le markdown de la carte affiché ressemblera à ça:

```
---
language: en
license: mit
library_name: keras
---

# Carte de modèle pour MyCoolModel

Ce modèle fait ceci et cela.

Ce modèle a été créé par [@nateraw](https://hf.co/nateraw).
```

Si vous mettez à jour n'importe quelle card.data, elle sera aussi
modifiée dans la carte elle même. 

```
card.data.library_name = 'timm'
card.data.language = 'fr'
card.data.license = 'apache-2.0'
print(card)
```

Maintenant, comme vous pouvez le voir, le header de métadonnée
a été mis à jour:

```
---
language: fr
license: apache-2.0
library_name: timm
---

# Carte de modèle pour MyCoolModel

Ce modèle peut faire ceci et cela...

Ce modèle a été créé par [@nateraw](https://hf.co/nateraw).
```

Tout en mettant à jour la donnée de carte, vous pouvez vérifier que la carte est toujours valide pour le Hub en appelant [`ModelCard.validate`]. Ceci vous assure que la carte passera n'importe quelle règle de validation existante sur le Hub Hugging Face.

### Depuis le template par défaut

Aulieu d'utiliser votre propre template, vous pouvez aussi utiliser le [template par défaut](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md), qui est une carte de modèle avec toutes les fonctionnalités possibles contenant des tonnes de sections que vous aurez peut-être besoin de remplir. En arrière plan, ce template utilise [Jinja2](https://jinja.palletsprojects.com/en/3.1.x/) pour remplir un fichier de template.

<Tip>

Notez que vous aurez besoin d'avoir Jinja2 installé pour utiliser `from_template`. Vous pouvez le faire avec
`pip install Jinja2`.

</Tip>

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

## Partagez une carte de modèle

Si vous êtes authentifié dans le Hub Hugging Face (soit en utilisant `huggingface-cli login` ou [`login`]), vous pouvez push des cartes vers le Hub
en appelant [`ModelCard.push_to_hub`]. Regardons comment le faire:

Tout d'abord, nous allons créer un nouveau dépôt qu'on appelera 'hf-hub-modelcards-pr-test' sur le namespace
de l'utilisateur authentifié:

```python
from huggingface_hub import whoami, create_repo

user = whoami()['name']
repo_id = f'{user}/hf-hub-modelcards-pr-test'
url = create_repo(repo_id, exist_ok=True)
```

Ensuite, nous créerons la carte pour le template par défaut (de la même manière que celui défini dans la section ci-dessus):

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

Enfin, nous pushong le tout sur le Hub

```python
card.push_to_hub(repo_id)
```

Vous pouvez vérifier la carte créé [ici](https://huggingface.co/nateraw/hf-hub-modelcards-pr-test/blob/main/README.md).

Si vous avez envie de faire une pull request, vous pouvez juste préciser `create_pr=True` lors de l'appel
`push_to_hub`:

```python
card.push_to_hub(repo_id, create_pr=True)
```

Une pull request créé de cette commande peut-être vue [ici](https://huggingface.co/nateraw/hf-hub-modelcards-pr-test/discussions/3).

## Mettre à jour les métadonnées

Dans cette section, nous verons ce que les métadonnées sont dans les cartes de dépôt
et comment les mettre à jour.

`metadata` fait référence à un contexte de hash map (ou clé-valeur) qui fournit des informations haut niveau sur un modèle, un dataset ou un espace. Cette information peut inclure des détails tels que le `type de pipeline`, le `model_id` ou le `model_description`. Pour plus de détails, vous pouvez consulter ces guides: [carte de modèle](https://huggingface.co/docs/hub/model-cards#model-card-metadata), [carte de dataset](https://huggingface.co/docs/hub/datasets-cards#dataset-card-metadata) and [Spaces Settings](https://huggingface.co/docs/hub/spaces-settings#spaces-settings). Maintenant voyons des exemples de mise à jour de ces métadonnées.


Commençons avec un premier exemple:

```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("username/my-cool-model", {"pipeline_tag": "image-classification"})
```

Avec ces deux lignes de code vous mettez à jour la métadonnée pour définir un nouveau `pipeline_tag`.

Par défaut, vous ne pouvez pas mettre à jour une clé qui existe déjà sur la carte. Si vous voulez le faire,
vous devez passer explicitement `overwrite=True`:


```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("username/my-cool-model", {"pipeline_tag": "text-generation"}, overwrite=True)
```

Souvent, vous aurez envie de suggérer des changements dans un dépôt sur
lequel vous avez pas les permissions d'écriture. Vous pouvez faire ainsi
en créant une pull request sur ce dépôt qui permettra aux propriétaires
de review et de fusionner vos suggestions.

```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("someone/model", {"pipeline_tag": "text-classification"}, create_pr=True)
```

## Inclure des résultats d'évaluation

Pour inclure des résultats d'évaluation dans la métadonnée `model-index`, vous pouvez passer un [`EvalResult`] ou une liste d'`EvalResult` avec vos résultats d'évaluation associés. En arrière-plan, le `model-index` sera créé lors de l'appel de `card.data.to_dict()`. Pour plus d'informations sur la manière dont tout ça fonctionne, vous pouvez consulter [cette section de la documentation du Hub](https://huggingface.co/docs/hub/models-cards#evaluation-results).

<Tip>

Notez qu'utiliser cette fonction vous demande d'inclure l'attribut `model_name` dans [`ModelCardData`].

</Tip>

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

Le `card.data` résultant devrait ressembler à ceci:

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

Si vous avez plus d'un résultat d'évaluation que vous voulez partager, passez simplement une liste
d'`EvalResult`:

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

Ce qui devrait donner le `card.data` suivant:

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