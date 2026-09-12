<!--⚠️ 请注意，此文件为 Markdown 格式，但包含我们文档生成器的特定语法（类似于 MDX），可能无法在您的 Markdown 查看器中正确渲染。
-->

# 创建与分享模型卡片

`huggingface_hub` 库提供了一套 Python 接口，用于创建、分享和更新模型卡片。
若想深入了解 Hub 上的模型卡片是什么、底层如何运作，请查阅[专门的文档页面](https://huggingface.co/docs/hub/models-cards)。

## 从 Hub 加载模型卡片

要加载 Hub 上已有的模型卡片，可以使用 [`ModelCard.load`] 函数。下面我们来加载 [`nateraw/vit-base-beans`](https://huggingface.co/nateraw/vit-base-beans) 的卡片。

```python
from huggingface_hub import ModelCard

card = ModelCard.load('nateraw/vit-base-beans')
```

这张卡片有几个您可能会用到的实用属性：
  - `card.data`：返回一个 [`ModelCardData`] 实例，包含模型卡片的元数据。对该实例调用 `.to_dict()` 可获得字典形式的结果。
  - `card.text`：返回卡片的文本内容，**不包含元数据头部**。
  - `card.content`：返回卡片的文本内容，**包含元数据头部**。

## 创建模型卡片

### 从文本创建

要从文本初始化模型卡片，只需在实例化 `ModelCard` 时传入卡片的文本内容即可。

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

另一种做法是使用 f-string。在下面的例子中，我们将：

- 使用 [`ModelCardData.to_yaml`] 把定义好的元数据转成 YAML，以便插入模型卡片的 YAML 块中。
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

上面的例子最终会得到这样一张卡片：

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

如果安装了 `Jinja2`，您还可以从 jinja 模板文件创建模型卡片。来看一个简单的例子：

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

生成的卡片 markdown 如下所示：

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

如果您修改了 `card.data` 中的任何内容，也会反映到卡片本身。

```
card.data.library_name = 'timm'
card.data.language = 'fr'
card.data.license = 'apache-2.0'
print(card)
```

可以看到，元数据头部已经更新：

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

在更新卡片数据的过程中，您可以调用 [`ModelCard.validate`] 来校验卡片对 Hub 而言是否仍然合法。这能确保卡片通过了 Hugging Face Hub 上设置的各项校验规则。

### 从默认模板创建

除了使用自己的模板，您也可以使用[默认模板](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md)——它是一份功能完整的模型卡片，包含大量供您填写的章节。其底层使用 [Jinja2](https://jinja.palletsprojects.com/en/3.1.x/) 来填充模板文件。

> [!TIP]
> 请注意，使用 `from_template` 需要安装 Jinja2，可通过 `pip install Jinja2` 安装。

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

如果您已通过 Hugging Face Hub 的身份认证（通过 `hf auth login` 或 [`login`]），只需调用 [`ModelCard.push_to_hub`] 就能把卡片推送到 Hub。下面看看具体怎么做……

首先，在已认证用户的命名空间下创建一个名为 'hf-hub-modelcards-pr-test' 的新仓库：

```python
from huggingface_hub import whoami, create_repo

user = whoami()['name']
repo_id = f'{user}/hf-hub-modelcards-pr-test'
url = create_repo(repo_id, exist_ok=True)
```

然后，用默认模板创建一张卡片（与上一节定义的相同）：

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

最后，把它推送到 Hub：

```python
card.push_to_hub(repo_id)
```

您可以在[这里](https://huggingface.co/nateraw/hf-hub-modelcards-pr-test/blob/main/README.md)查看生成的卡片。

如果您希望以 pull request 的形式推送卡片，只需在调用 `push_to_hub` 时指定 `create_pr=True`：

```python
card.push_to_hub(repo_id, create_pr=True)
```

由该命令创建的 PR 示例可参见[此处](https://huggingface.co/nateraw/hf-hub-modelcards-pr-test/discussions/3)。

## 更新元数据

在本节中，我们将了解仓库卡片中包含哪些元数据，以及如何更新它们。

`metadata`（元数据）指的是一组哈希映射（键值对）信息，用于描述模型、数据集或 Space 的高层信息。这些信息可以包括模型的 `pipeline type`、`model_id` 或 `model_description` 等细节。更多详情可参阅以下指南：[Model Card](https://huggingface.co/docs/hub/model-cards#model-card-metadata)、[Dataset Card](https://huggingface.co/docs/hub/datasets-cards#dataset-card-metadata) 和 [Spaces Settings](https://huggingface.co/docs/hub/spaces-settings#spaces-settings)。
下面我们来看几个更新这些元数据的例子。


先看第一个例子：

```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("username/my-cool-model", {"pipeline_tag": "image-classification"})
```

这两行代码会更新元数据，设置新的 `pipeline_tag`。

默认情况下，您无法更新卡片上已存在的键。如果确实需要，必须显式传入
`overwrite=True`：


```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("username/my-cool-model", {"pipeline_tag": "text-generation"}, overwrite=True)
```

有时您想对没有写入权限的仓库提出修改建议。这时可以在该仓库上创建一个 PR，让仓库所有者审阅并合并您的建议。

```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("someone/model", {"pipeline_tag": "text-classification"}, create_pr=True)
```

## 包含评估结果

要在元数据的 `model-index` 中包含评估结果，可以传入一个 [`EvalResult`] 或由多个 `EvalResult` 组成的列表。在调用 `card.data.to_dict()` 时，它会在底层生成 `model-index`。想了解其工作原理，您可以查看 [Hub 文档中的这一节](https://huggingface.co/docs/hub/models-cards#evaluation-results)。

> [!TIP]
> 请注意，使用此函数时需要在 [`ModelCardData`] 中包含 `model_name` 属性。

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

如果您要分享多个评估结果，只需传入一个 `EvalResult` 列表：

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

最终会得到如下的 `card.data`：

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
