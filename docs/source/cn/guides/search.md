<!--⚠️ 请注意，此文件是 Markdown 格式，但包含我们文档构建器的特定语法（类似于 MDX），可能无法在您的 Markdown 查看器中正确渲染。
-->

# 搜索 Hub

在本教程中，您将学习如何使用 `huggingface_hub` 在 Hub 上搜索模型、数据集和Spaces。

## 如何列出仓库？

`huggingface_hub`库包括一个 HTTP 客户端 [`HfApi`]，用于与 Hub 交互。 除此之外，它还可以列出存储在 Hub 上的模型、数据集和Spaces：

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> models = api.list_models()
```

[`list_models`] 返回一个迭代器，包含存储在 Hub 上的模型。

同样，您可以使用 [`list_datasets`] 列出数据集，使用 [`list_spaces`] 列出 Spaces。

## 如何过滤仓库？

列出仓库是一个好开始，但现在您可能希望对搜索结果进行过滤。
列出时，可以使用多个属性来过滤结果，例如：
- `filter`
- `author`
- `search`
- ...

让我们看一个示例，获取所有在 Hub 上进行图像分类的模型，这些模型已在 imagenet 数据集上训练，并使用 PyTorch 运行。

```py
models = hf_api.list_models(
	task="image-classification",
	library="pytorch",
	trained_dataset="imagenet",
)
```

在过滤时，您还可以对模型进行排序，并仅获取前几个结果。例如，以下示例获取了 Hub 上下载量最多的前 5 个数据集：

```py
>>> list(list_datasets(sort="downloads", direction=-1, limit=5))
[DatasetInfo(
	id='argilla/databricks-dolly-15k-curated-en',
	author='argilla',
	sha='4dcd1dedbe148307a833c931b21ca456a1fc4281',
	last_modified=datetime.datetime(2023, 10, 2, 12, 32, 53, tzinfo=datetime.timezone.utc),
	private=False,
	downloads=8889377,
	(...)
```



如果您想要在Hub上探索可用的过滤器, 请在浏览器中访问 [models](https://huggingface.co/models) 和 [datasets](https://huggingface.co/datasets) 页面
，尝试不同的参数并查看URL中的值。
