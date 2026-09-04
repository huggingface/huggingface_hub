<!--⚠️ 请注意，此文件采用 Markdown 格式，但包含文档构建器使用的特殊语法（类似于 MDX），
可能无法在 Markdown 查看器中正确呈现。
-->

# 创建和分享模型卡片

`huggingface_hub` 库提供了用于创建、分享和更新模型卡片的 Python 接口。
访问[专门的文档页面](https://huggingface.co/docs/hub/models-cards)，
深入了解 Hub 上的模型卡片及其底层工作方式。

## 从 Hub 加载模型卡片

要从 Hub 加载现有卡片，可以使用 [`ModelCard.load`] 函数。这里，我们将加载 [`nateraw/vit-base-beans`](https://huggingface.co/nateraw/vit-base-beans) 中的卡片。

```python
from huggingface_hub import ModelCard

card = ModelCard.load('nateraw/vit-base-beans')
```

此卡片提供了一些实用属性，可按需访问和使用：
  - `card.data`：返回一个包含模型卡片元数据的 [`ModelCardData`] 实例。对该实例调用 `.to_dict()` 即可获得字典形式的数据。
  - `card.text`：返回卡片的文本，*不包含元数据标头*。
  - `card.content`：返回卡片的文本内容，*包含元数据标头*。

## 创建模型卡片

### 从文本创建

要使用文本初始化模型卡片，只需在初始化 `ModelCard` 时传入卡片的文本内容。

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

也可以使用 f-string 来创建模型卡片。在下面的示例中，我们将：

- 使用 [`ModelCardData.to_yaml`] 将定义的元数据转换为 YAML，以便把 YAML 块插入模型卡片。
- 演示如何通过 Python f-string 使用模板变量。

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

上述示例将生成如下所示的卡片：

```
---
language: en
license: mit
library: timm
---

# My Model Card

This model created by [@nateraw](https://github.com/nateraw)
```

### 从 Jinja 模板创建

如果安装了 `Jinja2`，就可以通过 Jinja 模板文件创建模型卡片。下面来看一个基本示例：

```python
from pathlib import Path

from huggingface_hub import ModelCard, ModelCardData

# Define your jinja template
template_text = """
---
{{ card_data }}
---

# Model Card for MyCoolModel

This model does this and that.

This model was created by [@{{ author }}](https://hf.co/{{author}}).
""".strip()

# Write the template to a file
Path('custom_template.md').write_text(template_text)

# Define card metadata
card_data = ModelCardData(language='en', license='mit', library_name='keras')

# Create card from template, passing it any jinja template variables you want.
# In our case, we'll pass author
card = ModelCard.from_template(card_data, template_path='custom_template.md', author='nateraw')
card.save('my_model_card_1.md')
print(card)
```

生成的卡片 Markdown 如下：

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

更新 card.data 的任何字段时，卡片本身也会随之更新。

```
card.data.library_name = 'timm'
card.data.language = 'fr'
card.data.license = 'apache-2.0'
print(card)
```

可以看到，元数据标头现已更新：

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

随着卡片数据的更新，可以调用 [`ModelCard.validate`] 检查卡片是否仍然有效。这样可以确保卡片通过 Hugging Face Hub 设置的所有验证规则。

### 从默认模板创建

除了使用自己的模板，也可以使用[默认模板](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md)。这是一个功能完备的模型卡片，其中包含许多可供填写的章节。它在底层使用 [Jinja2](https://jinja.palletsprojects.com/en/3.1.x/) 来填充模板文件。

> [!TIP]
> 请注意，必须安装 Jinja2 才能使用 `from_template`。可以运行 `pip install Jinja2` 进行安装。

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

## 分享模型卡片

如果已经通过 Hugging Face Hub 进行身份验证（使用 `hf auth login` 或 [`login`]），只需调用 [`ModelCard.push_to_hub`]，即可将卡片推送到 Hub。下面来看具体操作。

首先，在已验证身份的用户命名空间下创建一个名为 'hf-hub-modelcards-pr-test' 的新仓库：

```python
from huggingface_hub import whoami, create_repo

user = whoami()['name']
repo_id = f'{user}/hf-hub-modelcards-pr-test'
url = create_repo(repo_id, exist_ok=True)
```

然后，使用默认模板创建一张卡片（与上一节中定义的模板相同）：

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

最后，将卡片推送到 Hub：

```python
card.push_to_hub(repo_id)
```

可以在[这里](https://huggingface.co/nateraw/hf-hub-modelcards-pr-test/blob/main/README.md)查看生成的卡片。

如果希望改为以拉取请求的形式推送卡片，只需指定 `create_pr=True` 作为调用 `push_to_hub` 时的参数：

```python
card.push_to_hub(repo_id, create_pr=True)
```

可以在[这里](https://huggingface.co/nateraw/hf-hub-modelcards-pr-test/discussions/3)查看此命令创建的拉取请求。

## 更新元数据

本节将介绍仓库卡片中包含哪些元数据，以及如何更新这些元数据。

`metadata` 是一种哈希映射（即键值对）结构，用于提供模型、数据集或 Space 的一些概要信息。这些信息可以包括模型的 `pipeline type`、`model_id` 或 `model_description` 等详情。有关更多信息，请参阅以下指南：[模型卡片](https://huggingface.co/docs/hub/model-cards#model-card-metadata)、[数据集卡片](https://huggingface.co/docs/hub/datasets-cards#dataset-card-metadata)和 [Space 设置](https://huggingface.co/docs/hub/spaces-settings#spaces-settings)。
下面通过几个示例了解如何更新这些元数据。


先来看第一个示例：

```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("username/my-cool-model", {"pipeline_tag": "image-classification"})
```

通过这两行代码，可以更新元数据并设置新的 `pipeline_tag`。

默认情况下，无法更新卡片中已经存在的键。如需更新，必须显式传入
`overwrite=True`：


```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("username/my-cool-model", {"pipeline_tag": "text-generation"}, overwrite=True)
```

通常会需要向自己没有写入权限的仓库建议一些更改。为此，可以在该仓库中创建拉取请求，让所有者审查并合并这些建议。

```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("someone/model", {"pipeline_tag": "text-classification"}, create_pr=True)
```

## 包含评估结果

要在元数据的 `model-index` 中包含评估结果，可以传入一个 [`EvalResult`]，也可以传入包含相关评估结果的 `EvalResult` 列表。调用 `card.data.to_dict()` 时，底层会生成 `model-index`。有关其工作方式的更多信息，请参阅 Hub 文档的[这一部分](https://huggingface.co/docs/hub/models-cards#evaluation-results)。

> [!TIP]
> 请注意，使用此功能时必须提供 `model_name` 属性；它需要包含在 [`ModelCardData`] 中。

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

生成的 `card.data` 应如下所示：

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

如果想要分享多个评估结果，只需传入 `EvalResult` 列表：

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

此时 `card.data` 将如下所示：

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
