<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Collections

Une collection est un groupe d'objets reliés entre eux sur le Hub (par exemple des modèles, des datases, des spaces ou des articles) qui sont organisés ensemble sur la même page. Les collections sont utiles pour créer votre propre portefeuille de contenu, mettre du contenu dans des catégories ou présenter une liste précise d'objets que vous voulez partager. Consultez ce [guide](https://huggingface.co/docs/hub/collections) pour comprendre en détail ce que sont les collections et ce à quoi elles ressemblent sur le Hub.

Vous pouvez gérer directement les collections depuis le navigateur, mais dans ce guide, nous nous concetrerons sur la gestion avec du code.

## Afficher une collection

Utiliser [`get_collection`] pour afficher vos collections ou n'importe quelle collection publique. Vous avez besoin du *slug* de la collection pour en récupérer une. Un slug est un identifiant pour une collection qui dépend du titre de l'ID de la collection. Vous pouvez trouver le slug dans l'URL de la page dédiée à la collection.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hfh_collection_slug.png"/>
</div>

Affichons la collection qui a pour slug `"TheBloke/recent-models-64f9a55bb3115b4f513ec026"`: 

```py
>>> from huggingface_hub import get_collection
>>> collection = get_collection("TheBloke/recent-models-64f9a55bb3115b4f513ec026")
>>> collection
Collection(
  slug='TheBloke/recent-models-64f9a55bb3115b4f513ec026',
  title='Recent models',
  owner='TheBloke',
  items=[...],
  last_updated=datetime.datetime(2023, 10, 2, 22, 56, 48, 632000, tzinfo=datetime.timezone.utc),
  position=1,
  private=False,
  theme='green',
  upvotes=90,
  description="Models I've recently quantized. Please note that currently this list has to be updated manually, and therefore is not guaranteed to be up-to-date."
)
>>> collection.items[0]
CollectionItem(
  item_object_id='651446103cd773a050bf64c2',
  item_id='TheBloke/U-Amethyst-20B-AWQ',
  item_type='model',
  position=88, 
  note=None
)
```

L'objet [`Collection`] retourné par [`get_collection`] contient:
- Des métadonnées: `slug`, `owner`, `title`, `description`, etc.
- Une liste d'objets [`CollectionItem`]; chaque objet représente un modèle, un dataset, un space ou un article.

Chaque objet d'une collection aura forcément:
- Un `item_object_id` unique: c'est l'id de l'objet de la collection dans la base de données
- Un `item_id`: c'est l'id dans le Hub de l'objet sous-jacent (modèle, dataset, space ou article); il n'est pas nécessairement unique, et seule la paire `item_id`/`item_type` sont uniques
- Un `item_type`: modèle, dataset, space ou article
- La `position` de l'objet dans la collection, qui peut-être mise à jour pour réorganiser votre collection (consultez [`update_collection_item`] ci dessous)

Une note peut aussi être attachées à un objet. Ceci permet d'ajouter des informations supplémentaire sur l'objet (un commentaire, un lien vers le post d'un blog, etc.). L'attribut a toujours une valeur `None` si un objet n'a pas de note.

En plus de ces attributs de base, les objets peuvent avoir des attributs supplémentaires en fonction de leur type: `author`, `private`, `lastModified`,`gated`, `title`, `likes`, `upvotes`, etc. Aucun de ces attribut ne sera retourné à coup sûr.

## Lister les collections

Nous pouvons aussi récupérer les collection en utilisant [`list_collections`]. Les collections peuvent être filtrées en utilisant certains paramètres. Listons toutes les collections de l'utilisateur [`teknium`](https://huggingface.co/teknium).
```py
>>> from huggingface_hub import list_collections

>>> collections = list_collections(owner="teknium")
```

Ce code renvoie un itérable d'objets `Collection`. On peut itérer sur ce dernier pour afficher, par exemple, le nombre d'upvote de chaque collection.

```py
>>> for collection in collections:
...   print("Number of upvotes:", collection.upvotes)
Number of upvotes: 1
Number of upvotes: 5
```

<Tip warning={true}>

Lorsque vous listez des collections, la liste d'objet est tronquée à 4 objets au maximum. Pour récupérer tous les objets d'une collection, vous devez utilisez [`get_collection`]

</Tip>

Il est possible d'avoir un filtrage plus avancé. Obtenons toutes les collections contenant le modèle [TheBloke/OpenHermes-2.5-Mistral-7B-GGUF](https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF), trié par popularité, et en limitant au nombre de 5, le nombre de collections affichées.

```py
>>> collections = list_collections(item="models/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF", sort="trending", limit=5):
>>> for collection in collections:
...   print(collection.slug)
teknium/quantized-models-6544690bb978e0b0f7328748
AmeerH/function-calling-65560a2565d7a6ef568527af
PostArchitekt/7bz-65479bb8c194936469697d8c
gnomealone/need-to-test-652007226c6ce4cdacf9c233
Crataco/favorite-7b-models-651944072b4fffcb41f8b568
```

Le paramètre `sort` doit prendre l'une des valeurs suivantes : `"last_modified"`,  `"trending"` ou `"upvotes"`. Le paramètre `item` prend
n'importe quel objet. Par exemple:
* `"models/teknium/OpenHermes-2.5-Mistral-7B"`
* `"spaces/julien-c/open-gpt-rhyming-robot"`
* `"datasets/squad"`
* `"papers/2311.12983"`

Pour plus de détails, consultez la référence à [`list_collections`]. 

## Créer une nouvelle collection

Maintenant que nous savons comment avoir une [`Collection`], créons la nôtre! Utilisez [`create_collection`] avec un titre et une description. Pour créer une collection sur la page d'une organisation, passez `namespace=mon_organisation` lors de la création de la collection. Enfin, vous pouvez aussi créer des collections privées en passant `private=True`

```py
>>> from huggingface_hub import create_collection

>>> collection = create_collection(
...     title="ICCV 2023",
...     description="Portefeuille de modèles, articles et démes présentées à l'ICCV 2023
... )
```

Un objet [`Collection`] sera retourné avec les métadonnées (titre, description, propriétaire, etc.) et une liste vide d'objets. Vous serez maintenant capable de vous référer à cette collection en utilisant son `slug`.
It will return a [`Collection`] object with the high-level metadata (title, description, owner, etc.) and an empty list of items. You will now be able to refer to this collection using it's `slug`.

```py
>>> collection.slug
'owner/iccv-2023-15e23b46cb98efca45'
>>> collection.title
"ICCV 2023"
>>> collection.owner
"username"
>>> collection.url
'https://huggingface.co/collections/owner/iccv-2023-15e23b46cb98efca45'
```

## Gérer des objets dans une collection

Maintenant que nous notre [`Collection`], nous allons y ajouter des objets et les organiser. 

### Ajouter des objets

Les objets doivent être ajoutés un par un en utilisant [`add_collection_item`]. Le seules données dont vous aurez besoin seront le `collection_slug`, l'`item_id` et l'`item_type`. En option, vous pouvez aussi ajouter un `note` à l'objet (500 caractères max).

```py
>>> from huggingface_hub import create_collection, add_collection_item

>>> collection = create_collection(title="OS Week Highlights - Sept 18 - 24", namespace="osanseviero")
>>> collection.slug
"osanseviero/os-week-highlights-sept-18-24-650bfed7f795a59f491afb80"

>>> add_collection_item(collection.slug, item_id="coqui/xtts", item_type="space")
>>> add_collection_item(
...     collection.slug,
...     item_id="warp-ai/wuerstchen",
...     item_type="model",
...     note="Würstchen is a new fast and efficient high resolution text-to-image architecture and model"
... )
>>> add_collection_item(collection.slug, item_id="lmsys/lmsys-chat-1m", item_type="dataset")
>>> add_collection_item(collection.slug, item_id="warp-ai/wuerstchen", item_type="space") # même item_id, mais l'item_type est différent
```

Si un objet existe déjà dans une collection (même paire `item_id`/`item_type`), une erreur HTTP 409 sera levée. Vous pouvez ignorer cette erreur en passant `exists_ok=True` dans la fonction.

### Ajouter une note à un objet de la collection

Vous pouvez modifier un objet existant pour ajouter ou changer la note attachée à l'objet en utilisant [`update_collection_item`].
Réutilisons l'exemple ci-dessus:

```py
>>> from huggingface_hub import get_collection, update_collection_item

# Récupére la collection avec les objets nouvellement ajoutés
>>> collection_slug = "osanseviero/os-week-highlights-sept-18-24-650bfed7f795a59f491afb80"
>>> collection = get_collection(collection_slug)

# Ajoute une note au dataset `lmsys-chat-1m`
>>> update_collection_item(
...     collection_slug=collection_slug,
...     item_object_id=collection.items[2].item_object_id,
...     note="This dataset contains one million real-world conversations with 25 state-of-the-art LLMs.",
... )
```

### Remettre en ordre les objets

Les objets dans une collection sont rangés dans un ordre. L'ordre est déterminé par l'attribut `position` de chacun des objets. Par défaut, les objets sont triés dans l'ordre d'ajout (du plus ancien au plus récent). Vous pouvez mettre à jour cet ordre en utilisant [`update_collection_item`] de la même manière que vous ajouteriez unr note

Réutilisons notre exemple ci-dessus:

```py
>>> from huggingface_hub import get_collection, update_collection_item

# Récupére la collection
>>> collection_slug = "osanseviero/os-week-highlights-sept-18-24-650bfed7f795a59f491afb80"
>>> collection = get_collection(collection_slug)

# Change l'ordre pour placer les deux objets `Wuerstchen` ensemble
>>> update_collection_item(
...     collection_slug=collection_slug,
...     item_object_id=collection.items[3].item_object_id,
...     position=2,
... )
```

### Supprimer des objets

Enfin, vous pouvez aussi supprimer un objet en utilisant [`delete_collection_item`].

```py
>>> from huggingface_hub import get_collection, update_collection_item

# Récupére la collection
>>> collection_slug = "osanseviero/os-week-highlights-sept-18-24-650bfed7f795a59f491afb80"
>>> collection = get_collection(collection_slug)

# Supprimer le space `coqui/xtts` de la liste
>>> delete_collection_item(collection_slug=collection_slug, item_object_id=collection.items[0].item_object_id)
```

## Supprimer une collection

Une collection peut être supprimée en utilisant [`delete_collection`].

<Tip warning={true}>

Cette action est irréversible. Une collection supprimée ne peut pas être restaurée.

</Tip>

```py
>>> from huggingface_hub import delete_collection
>>> collection = delete_collection("username/useless-collection-64f9a55bb3115b4f513ec026", missing_ok=True)
```