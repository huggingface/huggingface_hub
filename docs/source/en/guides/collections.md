<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Manage your collections

A collection is a group of related items on the Hub (models, datasets, Spaces, papers) that are organized together on a same page. Collections can be useful in many use cases such as creating your own portfolio, bookmarking content in categories or presenting a curated list of items your want to share. Check out this [guide](https://huggingface.co/docs/hub/collections) to understand in more details what are Collections and how they look like on the Hub,

Managing collections can be done in the browser directly. In this guide, we will focus on how to do it programmatically using `huggingface_hub`.

## Fetch a collection

To fetch a collection, use [`get_collection`]. You can use it either on your own collections or any public one. To retrieve a collection, you must have its collection's `slug`. A slug is an identifier for a collection based on the title and a unique ID. You can find it in the URL of the collection page.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hfh_collection_slug.png"/>
</div>

Here the slug is `"TheBloke/recent-models-64f9a55bb3115b4f513ec026"`. Let's fetch the collection:

```py
>>> from huggingface_hub import get_collection
>>> collection = get_collection("TheBloke/recent-models-64f9a55bb3115b4f513ec026")
>>> collection
Collection: { 
  {'description': "Models I've recently quantized.',
   'items': [...],
   'last_updated': datetime.datetime(2023, 9, 21, 7, 26, 28, 57000, tzinfo=datetime.timezone.utc),
   'owner': 'TheBloke',
   'position': 1,
   'private': False,
   'slug': 'TheBloke/recent-models-64f9a55bb3115b4f513ec026',
   'theme': 'green',
   'title': 'Recent models'}
}
>>> collection.items[0]
CollectionItem: { 
  {'item_object_id': '6507f6d5423b46492ee1413e',
   'author': 'TheBloke',
   'item_id': 'TheBloke/TigerBot-70B-Chat-GPTQ',
   'item_type': 'model',
   'lastModified': '2023-09-19T12:55:21.000Z',
   'position': 0,
   'private': False,
   'repoType': 'model'
   (...)
  }
}
```

The [`Collection`] object returned by [`get_collection`] contains:
- high-level metadata: `slug`, `owner`, `title`, `description`, etc.
- a list of [`CollectionItem`] objects. Each item represents a model, a dataset, a Space or a paper.

All items of a collection are guaranteed to have:
- a unique `item_object_id`: this is the id of the collection item in the database
- an `item_id`: this is the id on the Hub of the underlying item (model, dataset, Space, paper). It is not necessarily unique! only the `item_id`/`item_type` pair is unique.
- an `item_type`: model, dataset, Space, paper.
- the `position` of the item in the collection. Position can be updated to re-organize your collection (see [`update_collection_item`] below)

A `note` can also be attached to the item. This is useful to add additional information about the item (e.g. a comment, a link to a blog post, etc.). If an item doesn't have a note, the attribute still exists with a `None` value.

In addition to these base attributes, returned items can have additional attributes depending on their type: `author`, `private`, `lastModified`, `gated`, `title`, `likes`, `upvotes`, etc. None of these attributes are guaranteed to be returned.

## Create a new collection

Now that we know how to get a [`Collection`], let's create our own! Use [`create_collection`] with a title and optionally a description.

```py
>>> from huggingface_hub import create_collection

>>> collection = create_collection(
...     title="ICCV 2023",
...     description="Portfolio of models, papers and demos I presented at ICCV 2023",
... )
```

It will return a [`Collection`] object with the high-level metadata (title, description, owner, etc.) and an empty list of items. You will now be able to refer to this collection using it's `slug`.

```py
>>> collection.slug
'iccv-2023-15e23b46cb98efca45'
>>> collection.title
"ICCV 2023"
>>> collection.owner
"username"
```

To create a collection on an organization page, pass `namespace="my-cool-org"` when creating the collection. Finally, you can also create private collections by passing `private=True`.

## Manage items in a collection

Now that we have a [`Collection`], we want to add items to it and organize them.

### Add items

Items have to be added one by one using [`add_collection_item`]. You only need to know the `collection_slug`, `item_id` and `item_type`. Optionally, you can also add a `note` to the item (500 characters maximum).

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
>>> add_collection_item(collection.slug, item_id="warp-ai/wuerstchen", item_type="space") # same item_id, different item_type
```

If an item already exists in a collection (i.e. same `item_id`/`item_type` pair), an HTTP 409 error will be raised. You can choose to ignore this error by setting `exists_ok=True`.

### Add a note to an existing item

You can modify an existing item to add or modify the note attached to it using [`update_collection_item`]. Let's reuse the example above:

```py
>>> from huggingface_hub import get_collection, update_collection_item

# Fetch collection with newly added items
>>> collection_slug = "osanseviero/os-week-highlights-sept-18-24-650bfed7f795a59f491afb80"
>>> collection = get_collection(collection_slug)

# Add note the `lmsys-chat-1m` dataset
>>> update_collection_item(
...     collection_slug=collection_slug,
...     item_object_id=collection.items[2].item_object_id,
...     note="This dataset contains one million real-world conversations with 25 state-of-the-art LLMs.",
... )
```

### Re-order items

Items in a collection are ordered. The order is determined by the `position` attribute of each item. By default, items are ordered by appending new items at the end of the collection. You can update the ordering using [`update_collection_item`] the same way you would add a note.

Let's reuse our example above:

```py
>>> from huggingface_hub import get_collection, update_collection_item

# Fetch collection
>>> collection_slug = "osanseviero/os-week-highlights-sept-18-24-650bfed7f795a59f491afb80"
>>> collection = get_collection(collection_slug)

# Reorder to place the two `Wuerstchen` items together
>>> update_collection_item(
...     collection_slug=collection_slug,
...     item_object_id=collection.items[3].item_object_id,
...     position=2,
... )
```

### Remove items

Finally you can also remove an item using [`delete_collection_item`].

```py
>>> from huggingface_hub import get_collection, update_collection_item

# Fetch collection
>>> collection_slug = "osanseviero/os-week-highlights-sept-18-24-650bfed7f795a59f491afb80"
>>> collection = get_collection(collection_slug)

# Remove `coqui/xtts` Space from the list
>>> delete_collection_item(collection_slug=collection_slug, item_object_id=collection.items[0].item_object_id)
```

## Delete collection

A collection can be deleted using [`delete_collection`].

<Tip warning={true}>

This is a non-revertible action. A deleted collection cannot be restored.

</Tip>

```py
>>> from huggingface_hub import delete_collection
>>> collection = delete_collection("username/useless-collection-64f9a55bb3115b4f513ec026", missing_ok=True)
```