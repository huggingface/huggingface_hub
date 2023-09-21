<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Manage your Collections

A Collection is a group of related items from the Hub (models, datasets, Spaces, papers) that are organized together on a same page. They can be useful in many use cases such as creating your own portfolio, bookmarking content in categories or presenting a curated list of items your want to share. Check out this [guide](https://huggingface.co/docs/hub/collections) to understand in more details what are Collections and how they look like on the Hub, 

Managing Collections can be done in the browser directly. In this guide, we will focus on how to it programmatically using `huggingface_hub`.

## Get Collection content

To get the content of a collection, use [`get_collection`]. You can use it either on your own Collections or any public Collection. You retrieve a Collection, you must have its collection `slug`. A slug is a unique identifier for a collection based on the title and an ID. You can find it in the URL of the Collection page.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hfh_collection_slug.png"/>
</div>


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
  {'_id': '6507f6d5423b46492ee1413e',
   'author': 'TheBloke',
   'id': 'TheBloke/TigerBot-70B-Chat-GPTQ',
   'item_type': 'model',
   'lastModified': '2023-09-19T12:55:21.000Z',
   'position': 0,
   'private': False,
   'repoType': 'model'
   (...)
  }
}
```

## Create a new Collection

To create a collection, use [`create_collection`] with title and optionally a description.

```py
>>> from huggingface_hub import create_collection

>>> collection = create_collection(
...     title="ICCV 2023",
...     description="Portfolio of models, papers and demos I presented at ICCV 2023",
... )
```

It will return a [`Collection`] object with some high-level metadata (title, description, owner, etc.) and a list of items (currently empty). You will now be able to refer to this collection using it's `slug`.

```py
>>> collection.slug
'iccv-2023-15ecb98efca45'
>>> collection.title
"ICCV 2023"
```

## Add items to your Collection
