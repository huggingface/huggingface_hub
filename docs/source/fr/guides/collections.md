<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Collections

Une collection est un groupe d'éléments liés sur le Hub (modèles, datasets, Spaces, papers) qui sont organisés ensemble sur la même page. Les collections sont utiles pour créer votre propre portfolio, marquer du contenu par catégories, ou présenter une liste sélectionnée d'éléments que vous souhaitez partager. Consultez ce [guide](https://huggingface.co/docs/hub/collections) pour comprendre plus en détail ce que sont les collections et à quoi elles ressemblent sur le Hub.

Vous pouvez gérer les collections directement dans le navigateur, mais dans ce guide, nous nous concentrerons sur comment les gérer avec du code Python.

## Récupérer une collection

Utilisez [`get_collection`] pour récupérer vos collections ou toute collection publique. Vous devez avoir le *slug* de la collection pour la récupérer. Un slug est un identifiant pour une collection basé sur le titre et un ID unique. Vous pouvez trouver le slug dans l'URL de la page de la collection.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hfh_collection_slug.png"/>
</div>

Récupérons la collection avec `"TheBloke/recent-models-64f9a55bb3115b4f513ec026"` :

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

L'objet [`Collection`] retourné par [`get_collection`] contient :
- des métadonnées de haut niveau : `slug`, `owner`, `title`, `description`, etc.
- une liste d'objets [`CollectionItem`] ; chaque élément représente un modèle, un dataset, un Space ou un paper.

Tous les éléments de collection sont garantis d'avoir :
- un `item_object_id` unique : c'est l'id de l'élément de collection dans la base de données
- un `item_id` : c'est l'id sur le Hub de l'élément sous-jacent (modèle, dataset, Space, paper) ; il n'est pas nécessairement unique, et seule la paire `item_id`/`item_type` est unique
- un `item_type` : model, dataset, Space, paper
- la `position` de l'élément dans la collection, qui peut être mise à jour pour réorganiser votre collection (voir [`update_collection_item`] ci-dessous)

Une `note` peut également être attachée à l'élément. C'est utile pour ajouter des informations supplémentaires sur l'élément (un commentaire, un lien vers un article de blog, etc.). L'attribut a toujours une valeur `None` si un élément n'a pas de note.

En plus de ces attributs de base, les éléments retournés peuvent avoir des attributs supplémentaires selon leur type : `author`, `private`, `lastModified`, `gated`, `title`, `likes`, `upvotes`, etc. Aucun de ces attributs n'est garanti d'être retourné.

## Lister les collections

Nous pouvons également récupérer des collections en utilisant [`list_collections`]. Les collections peuvent être filtrées en utilisant certains paramètres. Listons toutes les collections de l'utilisateur [`teknium`](https://huggingface.co/teknium).

```py
>>> from huggingface_hub import list_collections

>>> collections = list_collections(owner="teknium")
```

Cela retourne un itérable d'objets `Collection`. Nous pouvons itérer dessus pour afficher, par exemple, le nombre d'upvotes pour chaque collection.

```py
>>> for collection in collections:
...   print("Number of upvotes:", collection.upvotes)
Number of upvotes: 1
Number of upvotes: 5
```

> [!WARNING]
> Lors du listage des collections, la liste d'éléments par collection est tronquée à 4 éléments maximum. Pour récupérer tous les éléments d'une collection, vous devez utiliser [`get_collection`].

Il est possible de faire un filtrage plus avancé. Obtenons toutes les collections contenant le modèle [TheBloke/OpenHermes-2.5-Mistral-7B-GGUF](https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF), triées par tendance, et limitons le nombre à 5.

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

Le paramètre `sort` doit être l'un des suivants : `"last_modified"`, `"trending"` ou `"upvotes"`. Le paramètre `item` accepte n'importe quel élément particulier. Par exemple :
* `"models/teknium/OpenHermes-2.5-Mistral-7B"`
* `"spaces/julien-c/open-gpt-rhyming-robot"`
* `"datasets/squad"`
* `"papers/2311.12983"`

Pour plus de détails, veuillez consulter la référence [`list_collections`].

## Créer une nouvelle collection

Maintenant que nous savons comment obtenir une [`Collection`], créons la nôtre ! Utilisez [`create_collection`] avec un titre et une description. Pour créer une collection sur la page d'une organisation, passez `namespace="my-cool-org"` lors de la création de la collection. Enfin, vous pouvez également créer des collections privées en passant `private=True`.

```py
>>> from huggingface_hub import create_collection

>>> collection = create_collection(
...     title="ICCV 2023",
...     description="Portfolio of models, papers and demos I presented at ICCV 2023",
... )
```

Cela retournera un objet [`Collection`] avec les métadonnées de haut niveau (titre, description, propriétaire, etc.) et une liste vide d'éléments. Vous pourrez maintenant vous référer à cette collection en utilisant son `slug`.

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

## Gérer les éléments dans une collection

Maintenant que nous avons une [`Collection`], nous voulons y ajouter des éléments et les organiser.

### Ajouter des éléments

Les éléments doivent être ajoutés un par un en utilisant [`add_collection_item`]. Vous avez seulement besoin de connaître le `collection_slug`, `item_id` et `item_type`. Optionnellement, vous pouvez également ajouter une `note` à l'élément (500 caractères maximum).

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
>>> add_collection_item(collection.slug, item_id="warp-ai/wuerstchen", item_type="space") # même item_id, item_type différent
```

Si un élément existe déjà dans une collection (même paire `item_id`/`item_type`), une erreur HTTP 409 sera levée. Vous pouvez choisir d'ignorer cette erreur en définissant `exists_ok=True`.

### Ajouter une note à un élément existant

Vous pouvez modifier un élément existant pour ajouter ou modifier la note qui lui est attachée en utilisant [`update_collection_item`]. Réutilisons l'exemple ci-dessus :

```py
>>> from huggingface_hub import get_collection, update_collection_item

# Récupérer la collection avec les éléments nouvellement ajoutés
>>> collection_slug = "osanseviero/os-week-highlights-sept-18-24-650bfed7f795a59f491afb80"
>>> collection = get_collection(collection_slug)

# Ajouter une note au dataset `lmsys-chat-1m`
>>> update_collection_item(
...     collection_slug=collection_slug,
...     item_object_id=collection.items[2].item_object_id,
...     note="This dataset contains one million real-world conversations with 25 state-of-the-art LLMs.",
... )
```

### Réorganiser les éléments

Les éléments dans une collection sont ordonnés. L'ordre est déterminé par l'attribut `position` de chaque élément. Par défaut, les éléments sont ordonnés en ajoutant les nouveaux éléments à la fin de la collection. Vous pouvez mettre à jour l'ordre en utilisant [`update_collection_item`] de la même manière que vous ajouteriez une note.

Réutilisons notre exemple ci-dessus :

```py
>>> from huggingface_hub import get_collection, update_collection_item

# Récupérer la collection
>>> collection_slug = "osanseviero/os-week-highlights-sept-18-24-650bfed7f795a59f491afb80"
>>> collection = get_collection(collection_slug)

# Réorganiser pour placer les deux éléments `Wuerstchen` ensemble
>>> update_collection_item(
...     collection_slug=collection_slug,
...     item_object_id=collection.items[3].item_object_id,
...     position=2,
... )
```

### Supprimer des éléments

Enfin, vous pouvez également supprimer un élément en utilisant [`delete_collection_item`].

```py
>>> from huggingface_hub import get_collection, delete_collection_item

# Récupérer la collection
>>> collection_slug = "osanseviero/os-week-highlights-sept-18-24-650bfed7f795a59f491afb80"
>>> collection = get_collection(collection_slug)

# Supprimer le Space `coqui/xtts` de la liste
>>> delete_collection_item(collection_slug=collection_slug, item_object_id=collection.items[0].item_object_id)
```

## Supprimer une collection

Une collection peut être supprimée en utilisant [`delete_collection`].

> [!WARNING]
> C'est une action non réversible. Une collection supprimée ne peut pas être restaurée.

```py
>>> from huggingface_hub import delete_collection
>>> collection = delete_collection("username/useless-collection-64f9a55bb3115b4f513ec026", missing_ok=True)
```
