<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Hub'da arama

Bu rehberde `huggingface_hub` kullanarak Hub'da modelleri, dataset'leri ve Space'leri nasıl arayacağını öğreneceksin.

## Depolar nasıl listelenir?

`huggingface_hub` kütüphanesi, Hub ile etkileşim kurmak için bir HTTP istemcisi olan [`HfApi`]'yi içerir.
Diğer şeylerin yanı sıra, Hub'da depolanan modelleri, dataset'leri ve Space'leri listeleyebilir:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> models = api.list_models()
```

[`list_models`]'ın çıktısı, Hub'da depolanan modeller üzerinde bir iterator'dır.

Benzer şekilde, dataset'leri listelemek için [`list_datasets`], Space'leri listelemek için ise [`list_spaces`] kullanabilirsin.

## Depolar nasıl filtrelenir?

Depoları listelemek harika ama şimdi aramanı filtrelemek isteyebilirsin.
Liste yardımcılarının birkaç özniteliği vardır, örneğin:
- `filter`
- `author`
- `search`
- `num_parameters`
- ...

Hub'da görüntü sınıflandırması yapan, imagenet dataset'i üzerinde eğitilmiş ve PyTorch ile çalışan tüm modelleri almak için bir örneğe bakalım.

```py
models = hf_api.list_models(filter=["image-classification", "pytorch", "imagenet"])
```

Modelleri parametre sayısına göre de Hub arayüzündekiyle aynı aralık sözdizimini kullanarak filtreleyebilirsin:

```py
models = hf_api.list_models(num_parameters="min:6B,max:128B")
```

Filtreleme yaparken modelleri sıralayıp yalnızca en üst sonuçları da alabilirsin. Örneğin,
aşağıdaki örnek Hub'daki en çok indirilen 5 dataset'i getirir:

```py
>>> list(list_datasets(sort="downloads", limit=5))
[DatasetInfo(
	id='argilla/databricks-dolly-15k-curated-en',
	author='argilla',
	sha='4dcd1dedbe148307a833c931b21ca456a1fc4281',
	last_modified=datetime.datetime(2023, 10, 2, 12, 32, 53, tzinfo=datetime.timezone.utc),
	private=False,
	downloads=8889377,
	(...)
```



Hub'daki kullanılabilir filtreleri keşfetmek için tarayıcında [models](https://huggingface.co/models) ve [datasets](https://huggingface.co/datasets) sayfalarını ziyaret et,
bazı parametreler için ara ve URL'deki değerlere bak.

## CLI kullanma

Modelleri, dataset'leri ve Space'leri `hf` komut satırı arayüzüyle de listeleyebilir ve arayabilirsin:

```bash
# List models
>>> hf models ls --search "llama" --sort downloads --limit 5

# List datasets
>>> hf datasets ls --author Qwen

# List Spaces
>>> hf spaces ls --search "3d"

# Get info about a specific model
>>> hf models info Lightricks/LTX-2
```

Daha fazla ayrıntı için [CLI rehberine](./cli.md#hf-models) bak.
