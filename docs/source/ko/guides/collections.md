<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->
# Collections[[collections]]

Collection은 Hub(모델, 데이터셋, Spaces, 논문)에 있는 관련 항목들의 그룹으로, 같은 페이지에 함께 구성되어 있습니다. Collections는 자신만의 포트폴리오를 만들거나, 카테고리별로 콘텐츠를 북마크 하거나, 공유하고 싶은 item들의 큐레이팅 된 목록을 제시하는 데 유용합니다. 여기 [가이드](https://huggingface.co/docs/hub/collections)를 확인하여 Collections가 무엇이고 Hub에서 어떻게 보이는지 자세히 알아보세요.

브라우저에서 직접 Collections를 관리할 수 있지만, 이 가이드에서는 프로그래밍 방식으로 Collection을 관리하는 방법에 초점을 맞추겠습니다.

## Collection 가져오기[[fetch-a-collection]]

[`get_collection`]을 사용하여 자신의 Collections나 공개된 Collection을 가져올 수 있습니다. Collection을 가져오려면 Collection의 *slug*가 필요합니다. Slug는 제목과 고유한 ID를 기반으로 한 Collection의 식별자입니다. Collection 페이지의 URL에서 slug를 찾을 수 있습니다.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hfh_collection_slug.png"/>
</div>

`"TheBloke/recent-models-64f9a55bb3115b4f513ec026"` Collection을 가져와 봅시다:

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

[`get_collection`]에 의해 반환된 [`Collection`] 객체에는 다음이 포함되어 있습니다:
- 높은 수준의 메타데이터: `slug`, `owner`, `title`, `description` 등
- [`CollectionItem`] 객체의 목록; 각 항목은 모델, 데이터셋, Space 또는 논문을 나타냅니다.

모든 Collection 항목에는 다음이 보장됩니다:
- 고유한 `item_object_id`: 데이터베이스에서 Collection 항목의 id
- 기본 항목(모델, 데이터셋, Space, 논문)의 Hub에서의 `item_id`; 고유하지 않으며, `item_id`/`item_type` 쌍만 고유합니다.
- `item_type`: 모델, 데이터셋, Space, 논문
- Collection에서 항목의 `position`으로, 이를 업데이트하여 Collection을 재구성할 수 있습니다(아래의 [`update_collection_item`] 참조)

각 항목에는 추가 정보(코멘트, 블로그 포스트 링크 등)를 위한 `note`도 첨부될 수 있습니다. 항목에 note가 없으면 해당 속성값은 `None`이 됩니다.

이러한 기본 속성 외에도, 반환된 항목은 유형에 따라 추가 속성(`author`, `private`, `lastModified`, `gated`, `title`, `likes`, `upvotes` 등)을 가질 수 있습니다. 그러나 이러한 속성이 반환된다는 보장은 없습니다.

## Collections 나열하기[[fetch-a-collection]]

[`list_collections`]를 사용하여 Collections를 나열할 수도 있습니다. Collections는 몇 가지 매개변수를 사용하여 필터링할 수 있습니다. 사용자 [`teknium`](https://huggingface.co/teknium)의 모든 Collections를 나열해 봅시다.

```py
>>> from huggingface_hub import list_collections

>>> collections = list_collections(owner="teknium")
```

이렇게 하면 `Collection` 객체의 반복 가능한 객체가 반환됩니다. 예를 들어 각 Collection의 upvotes 수를 출력하기 위해 반복할 수 있습니다.

```py
>>> for collection in collections:
...   print("Number of upvotes:", collection.upvotes)
Number of upvotes: 1
Number of upvotes: 5
```

<Tip warning={true}>

Collections를 나열할 때, 각 Collection의 항목 목록은 최대 4개 항목으로 잘립니다. Collection의 모든 항목을 가져오려면 [`get_collection`]을 사용해야 합니다.

</Tip>

고급 필터링을 수행할 수 있습니다. 예를 들어 모델 [TheBloke/OpenHermes-2.5-Mistral-7B-GGUF](https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF)를 포함하는 트렌딩 순으로 정렬된 Collections를 5개까지만 가져올 수 있습니다.

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

`sort` 매개변수는 `"last_modified"`, `"trending"` 또는 `"upvotes"` 중 하나여야 합니다. `item` 매개변수는 특정 항목을 받습니다. 예를 들면 다음과 같습니다:
* `"models/teknium/OpenHermes-2.5-Mistral-7B"`
* `"spaces/julien-c/open-gpt-rhyming-robot"`
* `"datasets/squad"`
* `"papers/2311.12983"`

자세한 내용은 [`list_collections`] 참조를 확인하시기 바랍니다.

## 새 Collection 만들기[[fetch-a-collection]]

이제 [`Collection`]을 가져오는 방법을 알았으니 우리만의 Collection을 만들어봅시다! 제목과 설명을 사용하여 [`create_collection`]을 호출합니다. 조직 페이지에 Collection을 만들려면 Collection 생성 시 `namespace="my-cool-org"`를 전달합니다. 마지막으로 `private=True`를 전달하여 비공개 Collection을 만들 수도 있습니다.

```py
>>> from huggingface_hub import create_collection

>>> collection = create_collection(
...     title="ICCV 2023",
...     description="Portfolio of models, papers and demos I presented at ICCV 2023",
... )
```

이렇게 하면 (제목, 설명, 소유자 등의) 높은 수준의 메타데이터와 빈 항목 목록을 가진 [`Collection`] 객체가 반환됩니다. 이제 `slug`를 사용하여 이 Collection을 참조할 수 있습니다.

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

## Collection의 item 관리[[manage-items-in-a-collection]]

이제 [`Collection`]을 가지고 있으므로, 여기에 item을 추가하고 구성해봅시다.

### item 추가[[add-items]]

item은 [`add_collection_item`]을 사용하여 하나씩 추가해야 합니다. `collection_slug`, `item_id`, `item_type`만 알면 됩니다. 또한 선택적으로 항목에 `note`를 추가할 수도 있습니다(최대 500자).

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
>>> add_collection_item(collection.slug, item_id="warp-ai/wuerstchen", item_type="space") # 동일한 item_id, 다른 item_type
```

Collection에 item이 이미 존재하는 경우(동일한 `item_id`/`item_type` 쌍), HTTP 409 오류가 발생합니다. `exists_ok=True`를 설정하면 이 오류를 무시할 수 있습니다.

### 기존 item에 메모 추가[[add-a-note-to-an-existing-item]]

[`update_collection_item`]을 사용하여 기존 item을 수정하여 메모를 추가하거나 변경할 수 있습니다. 위의 예시를 다시 사용해 봅시다:

```py
>>> from huggingface_hub import get_collection, update_collection_item

# 새로 추가된 item과 함께 Collection 가져오기
>>> collection_slug = "osanseviero/os-week-highlights-sept-18-24-650bfed7f795a59f491afb80"
>>> collection = get_collection(collection_slug)

# `lmsys-chat-1m` 데이터셋에 메모 추가
>>> update_collection_item(
...     collection_slug=collection_slug,
...     item_object_id=collection.items[2].item_object_id,
...     note="This dataset contains one million real-world conversations with 25 state-of-the-art LLMs.",
... )
```

### item 재정렬[[reorder-items]]

Collection의 item은 순서가 있습니다. 이 순서는 각 item의 `position` 속성에 의해 결정됩니다. 기본적으로 item은 Collection의 끝에 추가되는 방식으로 순서가 지정됩니다. [`update_collection_item`]을 사용하여 메모를 추가하는 것과 같은 방식으로 순서를 업데이트할 수 있습니다.

위의 예시를 다시 사용해 봅시다:

```py
>>> from huggingface_hub import get_collection, update_collection_item

# Collection 가져오기
>>> collection_slug = "osanseviero/os-week-highlights-sept-18-24-650bfed7f795a59f491afb80"
>>> collection = get_collection(collection_slug)

# 두 개의 `Wuerstchen` item을 함께 배치하도록 재정렬
>>> update_collection_item(
...     collection_slug=collection_slug,
...     item_object_id=collection.items[3].item_object_id,
...     position=2,
... )
```

### item 제거[[remove-items]]

마지막으로 [`delete_collection_item`]을 사용하여 item을 제거할 수도 있습니다.

```py
>>> from huggingface_hub import get_collection, update_collection_item

# Collection 가져오기
>>> collection_slug = "osanseviero/os-week-highlights-sept-18-24-650bfed7f795a59f491afb80"
>>> collection = get_collection(collection_slug)

# 목록에서 `coqui/xtts` Space 제거
>>> delete_collection_item(collection_slug=collection_slug, item_object_id=collection.items[0].item_object_id)
```

## Collection 삭제[[delete-collection]]

[`delete_collection`]을 사용하여 Collection을 삭제할 수 있습니다.

<Tip warning={true}>

이 작업은 되돌릴 수 없습니다. 삭제된 Collection은 복구할 수 없습니다.

</Tip>

```py
>>> from huggingface_hub import delete_collection
>>> collection = delete_collection("username/useless-collection-64f9a55bb3115b4f513ec026", missing_ok=True)
```