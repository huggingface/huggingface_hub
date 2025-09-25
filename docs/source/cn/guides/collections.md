<!--⚠️ 请注意，该文件使用Markdown格式，但包含我们的文档构建器的特定语法（类似与MDX),您的Markdown查看器可能无法正确渲染
-->

# 集合（Collections）

集合（collection）是 Hub 上将一组相关项目（模型、数据集、Spaces、论文）组织在同一页面上的一种方式。利用集合，你可以创建自己的作品集、为特定类别的内容添加书签，或呈现你想要分享的精选条目。要了解更多关于集合的概念及其在 Hub 上的呈现方式，请查看这篇 [指南](https://huggingface.co/docs/hub/collections) 

你可以直接在浏览器中管理集合，但本指南将重点介绍如何以编程方式进行管理。

## 获取集合

使用 [`get_collection`] 来获取你的集合或任意公共集合。 你需要提供集合的 *slug* 才能检索到该集合。 slug 是基于集合标题和唯一 ID 的标识符。你可以在集合页面的 URL 中找到它。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hfh_collection_slug.png"/>
</div>

让我们获取`"TheBloke/recent-models-64f9a55bb3115b4f513ec026"`这个集合:

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

[`get_collection`] 返回的 [`Collection`] 对象包含以下信息:
- 高级元数据: `slug`, `owner`, `title`, `description`等。
- 一个 [`CollectionItem`] 对象列表; 每个条目代表一个模型、数据集、Space 或论文。

所有集合条目（items）都保证具有：
- 唯一的 `item_object_id`: 这是集合条目在数据库中的唯一 ID
- 一个 `item_id`: Hub 上底层条目的 ID（模型、数据集、Space、论文）；此 ID 不一定是唯一的，仅当 `item_id` 与 `item_type` 成对出现时才唯一
- 一个 `item_type`: 如`model`, `dataset`, `Space`, `paper`
- 该条目在集合中的 `position`, 可通过后续操作 (参加下文的 [`update_collection_item`])来重新排序集合条目

此外，`note` 可选地附加在条目上。这对为某个条目添加额外信息（评论、博客文章链接等）很有帮助。如果条目没有备注，`note` 的值为 `None`。

除了这些基本属性之外，不同类型的条目可能会返回额外属性，如：`author`、`private`、`lastModified`、`gated`、`title`、`likes`、`upvotes` 等。这些属性不保证一定存在。

## 列出集合

我们也可以使用 [`list_collections`]来检索集合，并通过一些参数进行过滤。让我们列出用户[`teknium`](https://huggingface.co/teknium)的所有集合：
```py
>>> from huggingface_hub import list_collections

>>> collections = list_collections(owner="teknium")
```

这将返回一个 Collection 对象的可迭代序列。我们可以遍历它们，比如打印每个集合的点赞数（upvotes）：

```py
>>> for collection in collections:
...   print("Number of upvotes:", collection.upvotes)
Number of upvotes: 1
Number of upvotes: 5
```

> [!WARNING]
> 当列出集合时，每个集合中返回的条目列表最多会被截断为 4 个。若要检索集合中的所有条目，你必须使用 [`get_collection`].

我们可以进行更高级的过滤。例如，让我们获取所有包含模型 [TheBloke/OpenHermes-2.5-Mistral-7B-GGUF](https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF) 的集合，并按照趋势（trending）进行排序，同时将结果限制为 5 个。
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

`sort` 参数必须是 `"last_modified"`、`"trending"` 或 `"upvotes"` 之一。`item` 参数接受任意特定条目，例如：
* `"models/teknium/OpenHermes-2.5-Mistral-7B"`
* `"spaces/julien-c/open-gpt-rhyming-robot"`
* `"datasets/squad"`
* `"papers/2311.12983"`

详情请查看 [`list_collections`] 的参考文档。

## 创建新集合

现在我们已经知道如何获取一个 [`Collection`], 让我们自己创建一个吧！ 使用 [`create_collection`]，传入一个标题和描述即可。 若要在组织（organization）名下创建集合，可以通过 `namespace="my-cool-org"` 参数指定。同样，你也可以通过传入 `private=True` 创建私有集合。

```py
>>> from huggingface_hub import create_collection

>>> collection = create_collection(
...     title="ICCV 2023",
...     description="Portfolio of models, papers and demos I presented at ICCV 2023",
... )
```

该函数会返回一个包含高级元数据（标题、描述、所有者等）和空条目列表的 [`Collection`] 对象。现在你可以使用返回的 `slug` 来引用该集合。

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

## 管理集合中的条目

现在我们有了一个 [`Collection`]，接下来要添加条目并进行管理。

### 添加条目

使用 [`add_collection_item`] 来向集合中添加条目（一次添加一个）。你只需要提供 `collection_slug`、`item_id` 和 `item_type`。可选参数 `note` 用于为该条目添加附加说明（最多 500 个字符）。

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

如果一个条目已存在于集合中（相同的 `item_id` 和 `item_type`），将会引发 HTTP 409 错误。你可以通过设置 `exists_ok=True` 来忽略此错误。

### 为已存在条目添加备注

你可以使用 [`update_collection_item`] 来为已存在条目添加或修改备注。让我们重用上面的示例：

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

### 重新排序条目

集合中的条目是有序的。该顺序由每个条目的 `position` 属性决定。默认情况下，新添加的条目会被追加到集合末尾。你可以通过 [`update_collection_item`] 来更新顺序。

再次使用之前的示例：

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

### 删除条目

最后，你也可以使用 [`delete_collection_item`] 来删除集合中的条目。

```py
>>> from huggingface_hub import get_collection, update_collection_item

# Fetch collection
>>> collection_slug = "osanseviero/os-week-highlights-sept-18-24-650bfed7f795a59f491afb80"
>>> collection = get_collection(collection_slug)

# Remove `coqui/xtts` Space from the list
>>> delete_collection_item(collection_slug=collection_slug, item_object_id=collection.items[0].item_object_id)
```

## 删除集合

可以使用 [`delete_collection`] 来删除集合。

> [!WARNING]
> 此操作不可逆。删除的集合无法恢复。

```py
>>> from huggingface_hub import delete_collection
>>> collection = delete_collection("username/useless-collection-64f9a55bb3115b4f513ec026", missing_ok=True)
```
