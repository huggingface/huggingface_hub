<!--⚠️ 注意：此文件是 Markdown 格式，但包含文档生成器的特定语法（类似 MDX），可能无法在普通 Markdown 查看器中正确渲染。
-->

# 将任意机器学习框架集成到 Hub

Hugging Face Hub 让模型的托管和社区共享变得简单。它支持开源生态中的[几十个库](https://huggingface.co/docs/hub/models-libraries)。我们一直在努力扩展这项支持，推动机器学习协作向前发展。`huggingface_hub` 库在这个过程中发挥着重要作用，它允许任意 Python 脚本轻松推送和加载文件。

将库集成到 Hub 主要有四种方式：

1. **推送到 Hub：** 实现将模型上传到 Hub 的方法。这包括模型权重、[模型卡片](https://huggingface.co/docs/huggingface_hub/how-to-model-cards)，以及运行模型所需的其他相关信息或数据（例如训练日志）。这种方法通常称为 `push_to_hub()`。
2. **从 Hub 下载：** 实现从 Hub 加载模型的方法。该方法应下载模型配置和权重并加载模型。这种方法通常称为 `from_pretrained` 或 `load_from_hub()`。
3. **小组件：** 在 Hub 上模型的落地页中显示小组件，让用户可以直接在浏览器中快速试用模型。

本指南将重点介绍前两种方式，并介绍两种可用于集成库的主要方案及其优缺点。指南末尾会进行总结，帮助您在两种方案之间做出选择。请注意，这些只是可以根据自身需求调整的建议。

如果您对推理和小组件感兴趣，可以参考[这份指南](https://huggingface.co/docs/hub/models-adding-libraries#set-up-the-inference-api)。无论采用哪种方式，如果您正在将库集成到 Hub 并希望将其列入[我们的文档](https://huggingface.co/docs/hub/models-libraries)，都可以联系我们。

## 灵活的方案：辅助方法

将库集成到 Hub 的第一种方案，是自行实现 `push_to_hub` 和 `from_pretrained` 方法。这样可以完全控制需要上传或下载的文件，以及如何处理框架特有的输入。您可以参考[上传文件](./upload)和[下载文件](./download)指南，进一步了解具体做法。例如，FastAI 集成就是这样实现的（参见 [`push_to_hub_fastai`] 和 [`from_pretrained_fastai`]）。

不同库的实现可能有所差异，但工作流通常相似。

### from_pretrained

`from_pretrained` 方法通常如下所示：

```python
def from_pretrained(model_id: str) -> MyModelClass:
   # 从 Hub 下载模型
   cached_model = hf_hub_download(
      repo_id=repo_id,
      filename="model.pkl",
      library_name="fastai",
      library_version=get_fastai_version(),
   )

   # 加载模型
    return load_model(cached_model)
```

### push_to_hub

`push_to_hub` 方法通常需要处理仓库创建、生成模型卡片和保存权重，因此会稍微复杂一些。一种常见做法是将这些文件保存到临时文件夹中，上传后再删除临时文件夹。

```python
def push_to_hub(model: MyModelClass, repo_name: str) -> None:
   api = HfApi()

   # 如果仓库不存在则创建，并获取对应的 repo_id
   repo_id = api.create_repo(repo_name, exist_ok=True)

   # 将所有文件保存到临时目录，并在一次提交中推送
   with TemporaryDirectory() as tmpdir:
      tmpdir = Path(tmpdir)

      # 保存权重
      save_model(model, tmpdir / "model.safetensors")

      # 生成模型卡片
      card = generate_model_card(model)
      (tmpdir / "README.md").write_text(card)

      # 保存日志
      # 保存图表
      # 保存评估指标
      # ...

      # 推送到 Hub
      return api.upload_folder(repo_id=repo_id, folder_path=tmpdir)
```

当然，这只是一个示例。如果您需要进行更复杂的操作（删除远程文件、动态上传权重、在本地持久化权重等），请参考[上传文件](./upload)指南。

### 局限性

这种方案虽然灵活，但也有一些缺点，尤其是在维护方面。Hugging Face 用户在使用 `huggingface_hub` 时通常习惯了更多额外功能。例如，从 Hub 加载文件时，通常会提供以下参数：

- `token`：从私有仓库下载
- `revision`：从指定分支下载
- `cache_dir`：将文件缓存到指定目录
- `force_download`/`local_files_only`：控制是否复用缓存
- `proxies`：配置 HTTP 会话

推送模型时也支持类似的参数：

- `commit_message`：自定义提交信息
- `private`：如果仓库不存在则创建私有仓库
- `create_pr`：创建 PR，而不是推送到 `main`
- `branch`：推送到指定分支，而不是 `main` 分支
- `allow_patterns`/`ignore_patterns`：筛选要上传的文件
- `token`
- ...

上面看到的实现都可以添加这些参数，并将参数传递给 `huggingface_hub` 方法。不过，如果某个参数发生变化或新增功能，您就需要更新自己的包。支持这些参数也意味着需要在自己的侧维护更多文档。要了解如何缓解这些局限性，请继续阅读下一节的**类继承**方案。

## 更复杂的方案：类继承

如上所述，将库接入 Hub 主要需要包含两个方法：上传文件（`push_to_hub`）和下载文件（`from_pretrained`）。您可以自行实现这些方法，但这种做法存在一些问题。为了解决这些问题，`huggingface_hub` 提供了一个使用类继承的工具。下面来看看它是如何工作的！

在许多情况下，一个库已经使用 Python 类实现了模型。这个类包含模型的属性，以及加载、运行、训练和评估模型的方法。我们的方案是使用 mixin 扩展这个类，为其加入上传和下载功能。[Mixin](https://stackoverflow.com/a/547714) 是一种用于通过多重继承，为现有类扩展一组特定功能的类。`huggingface_hub` 提供了自己的 mixin，即 [`ModelHubMixin`]。关键在于理解它的行为以及如何对其进行自定义。

[`ModelHubMixin`] 类实现了 3 个*公开*方法（`push_to_hub`、`save_pretrained` 和 `from_pretrained`）。用户会调用这些方法来加载或保存模型。 [`ModelHubMixin`] 还定义了 2 个*私有*方法（`_save_pretrained` 和 `_from_pretrained`），这些方法需要由您实现。因此，要集成自己的库，您需要：

1. 让模型类继承 [`ModelHubMixin`]。
2. 实现私有方法：
   - [`~ModelHubMixin._save_pretrained`]：接收目录路径作为输入并将模型保存到其中的方法。您必须在此方法中编写转储模型的全部逻辑，包括模型卡片、模型权重、配置文件、训练日志和图表。与模型相关的所有信息都应由此方法处理。[模型卡片](https://huggingface.co/docs/hub/model-cards)对于描述模型尤其重要。更多详情请参阅[我们的实现指南](./model-cards)。
   - [`~ModelHubMixin._from_pretrained`]：接收 `model_id` 作为输入并返回实例化模型的**类方法**。该方法必须下载相关文件并加载这些文件。
3. 完成！

使用 [`ModelHubMixin`] 的优点是：只要处理好文件的序列化和加载，就可以直接使用了。您不需要担心仓库创建、提交、PR 或修订版本等问题。 [`ModelHubMixin`] 还会确保公开方法具有文档和类型注解，并且您可以在 Hub 上查看模型的下载次数。这些都由 [`ModelHubMixin`] 处理，并提供给您的用户使用。

### 一个具体示例：PyTorch

上面介绍的方案可以参考 [`PyTorchModelHubMixin`]，它是我们针对 PyTorch 框架提供的集成方案，可以直接使用。

#### 如何使用？

下面展示用户如何将 PyTorch 模型从 Hub 加载到本地，以及如何将模型保存并推送到 Hub：

```python
>>> import torch
>>> import torch.nn as nn
>>> from huggingface_hub import PyTorchModelHubMixin


# 按照平时的方式定义 PyTorch 模型
>>> class MyModel(
...         nn.Module,
...         PyTorchModelHubMixin, # 多重继承
...         library_name="keras-nlp",
...         tags=["keras"],
...         repo_url="https://github.com/keras-team/keras-nlp",
...         docs_url="https://keras.io/keras_nlp/",
...         # ^ 用于生成模型卡片的可选元数据
...     ):
...     def __init__(self, hidden_size: int = 512, vocab_size: int = 30000, output_size: int = 4):
...         super().__init__()
...         self.param = nn.Parameter(torch.rand(hidden_size, vocab_size))
...         self.linear = nn.Linear(output_size, vocab_size)

...     def forward(self, x):
...         return self.linear(x + self.param)

# 1. 创建模型
>>> model = MyModel(hidden_size=128)

# 配置会根据输入和默认值自动创建
>>> model.param.shape[0]
128

# 2.（可选）将模型保存到本地目录
>>> model.save_pretrained("path/to/my-awesome-model")

# 3. 将模型权重推送到 Hub
>>> model.push_to_hub("my-awesome-model")

# 4. 从 Hub 初始化模型 => 配置已保留
>>> model = MyModel.from_pretrained("username/my-awesome-model")
>>> model.param.shape[0]
128

# 模型卡片已正确填充
>>> from huggingface_hub import ModelCard
>>> card = ModelCard.load("username/my-awesome-model")
>>> card.data.tags
["keras", "pytorch_model_hub_mixin", "model_hub_mixin"]
>>> card.data.library_name
"keras-nlp"
```

#### 实现

实现其实非常直接，完整实现可以在[这里](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hub_mixin.py)找到。

1. 首先，让您的类继承 `ModelHubMixin`：

```python
from huggingface_hub import ModelHubMixin

class PyTorchModelHubMixin(ModelHubMixin):
   (...)
```

2. 实现 `_save_pretrained` 方法：

```py
from huggingface_hub import ModelHubMixin

class PyTorchModelHubMixin(ModelHubMixin):
   (...)

    def _save_pretrained(self, save_directory: Path) -> None:
        """将 PyTorch 模型的权重保存到本地目录。"""
        save_model_as_safetensor(self.module, str(save_directory / SAFETENSORS_SINGLE_FILE))

```

3. 实现 `_from_pretrained` 方法：

```py
class PyTorchModelHubMixin(ModelHubMixin):
   (...)

   @classmethod # 必须是类方法！
   def _from_pretrained(
       cls,
       *,
       model_id: str,
       revision: str,
       cache_dir: str,
       force_download: bool,
       local_files_only: bool,
       token: Union[str, bool, None],
       map_location: str = "cpu", # 额外参数
       strict: bool = False, # 额外参数
       **model_kwargs,
   ):
       """加载 PyTorch 预训练权重并返回已加载的模型。"""
         model = cls(**model_kwargs)
         if os.path.isdir(model_id):
             print("从本地目录加载权重")
             model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
             return cls._load_as_safetensor(model, model_file, map_location, strict)

          model_file = hf_hub_download(
             repo_id=model_id,
             filename=SAFETENSORS_SINGLE_FILE,
             revision=revision,
             cache_dir=cache_dir,
             force_download=force_download,
             token=token,
             local_files_only=local_files_only,
             )
          return cls._load_as_safetensor(model, model_file, map_location, strict)
```

就是这样！现在，您的库已经支持向 Hub 上传文件以及从 Hub 下载文件。

### 高级用法

上面一节简要介绍了 [`ModelHubMixin`] 的工作方式。本节将介绍一些高级功能，帮助您改进库与 Hugging Face Hub 的集成。

#### 模型卡片

[`ModelHubMixin`] 会为您生成模型卡片。模型卡片是随模型一起提供的文件，其中包含模型的重要信息。在底层，模型卡片是带有额外元数据的 Markdown 文件。模型卡片对于模型的可发现性、可复现性和共享至关重要！更多详情请参阅[模型卡片指南](https://huggingface.co/docs/hub/model-cards)。

半自动生成模型卡片是确保使用您的库推送的所有模型包含通用元数据的好方法，例如 `library_name`、`tags`、`license` 和 `pipeline_tag` 等。这样一来，您的库支持的所有模型都可以在 Hub 上轻松搜索，并且可以为访问模型页面的用户提供资源链接。继承 [`ModelHubMixin`] 时，可以直接定义这些元数据：

```py
class UniDepthV1(
   nn.Module,
   PyTorchModelHubMixin,
   library_name="unidepth",
   repo_url="https://github.com/lpiccinelli-eth/UniDepth",
   docs_url=...,
   pipeline_tag="depth-estimation",
   license="cc-by-nc-4.0",
   tags=["monocular-metric-depth-estimation", "arxiv:1234.56789"]
):
   ...
```

默认情况下，系统会根据您提供的信息生成一个通用模型卡片（例如：[pyp1/VoiceCraft_giga830M](https://huggingface.co/pyp1/VoiceCraft_giga830M)）。不过，您也可以定义自己的模型卡片模板！

在下面的示例中，使用 `VoiceCraft` 类推送的所有模型都会自动包含引用部分和许可证详情。有关如何定义模型卡片模板的更多信息，请参阅[模型卡片指南](./model-cards)。

```py
MODEL_CARD_TEMPLATE = """
---
# 有关模型卡片元数据的参考，请参阅规范：https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# 文档/指南：https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

这是一个 VoiceCraft 模型。更多详情请参阅官方 GitHub 仓库：https://github.com/jasonppy/VoiceCraft。此模型基于 Attribution-NonCommercial-ShareAlike 4.0 International 许可证共享。

## 引用

@article{peng2024voicecraft,
  author    = {Peng, Puyuan and Huang, Po-Yao and Li, Daniel and Mohamed, Abdelrahman and Harwath, David},
  title     = {VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild},
  journal   = {arXiv},
  year      = {2024},
}
"""

class VoiceCraft(
   nn.Module,
   PyTorchModelHubMixin,
   library_name="voicecraft",
   model_card_template=MODEL_CARD_TEMPLATE,
   ...
):
   ...
```

最后，如果您希望使用动态值扩展模型卡片生成过程，可以重写 [`~ModelHubMixin.generate_model_card`] 方法：

```py
from huggingface_hub import ModelCard, PyTorchModelHubMixin

class UniDepthV1(nn.Module, PyTorchModelHubMixin, ...):
   (...)

   def generate_model_card(self, *args, **kwargs) -> ModelCard:
      card = super().generate_model_card(*args, **kwargs)
      card.data.metrics = ...  # 添加指标
      card.text += ... # 追加章节
      return card
```

#### 配置

[`ModelHubMixin`] 会为您处理模型配置。它会在实例化模型时自动检查输入值，并将这些值序列化到 `config.json` 文件中。这带来两个好处：

1. 用户可以使用与您完全相同的参数重新加载模型。
2. 拥有 `config.json` 文件会自动启用 Hub 上的分析功能（例如“下载”次数统计）。

但它在实际中是如何工作的呢？以下几条规则可以让整个过程更符合用户预期：

- 如果您的 `__init__` 方法需要一个 `config` 输入，它会自动作为 `config.json` 保存到仓库中。
- 如果 `config` 输入参数标注了数据类类型（例如 `config: Optional[MyConfigClass] = None`），那么 `config` 值会被正确地反序列化。
- 初始化时传入的所有值也会存储在配置文件中。这意味着您不一定要显式需要一个 `config` 输入，仍然可以使用此功能。

示例：

```py
class MyModel(ModelHubMixin):
   def __init__(value: str, size: int = 3):
       self.value = value
       self.size = size

   (...) # 实现 _save_pretrained / _from_pretrained

model = MyModel(value="my_value")
model.save_pretrained(...)

# config.json 包含传入的值和默认值
{"value": "my_value", "size": 3}
```

但是，如果某个值无法序列化为 JSON，该怎么办？默认情况下，保存配置文件时会忽略这个值。不过，在某些情况下，您的库已经需要接收无法序列化的自定义对象作为输入，而您又不希望更新内部逻辑来修改其类型。别担心！继承 [`ModelHubMixin`] 时，可以为任意类型传入自定义编码器和解码器。虽然这需要多做一些工作，但能够确保在将库集成到 Hub 时不必改动内部逻辑。

下面是一个具体示例：某个类需要接收 `argparse.Namespace` 配置作为输入：

```py
class VoiceCraft(nn.Module):
    def __init__(self, args):
      self.pattern = self.args.pattern
      self.hidden_size = self.args.hidden_size
      ...
```

一种解决方案是将 `__init__` 签名更新为 `def __init__(self, pattern: str, hidden_size: int)`，并更新所有实例化类的代码片段。这是完全有效的修复方式，但可能会破坏使用您库的下游应用。

另一种方案是提供简单的编码器和解码器，将 `argparse.Namespace` 转换为字典。

```py
from argparse import Namespace

class VoiceCraft(
   nn.Module,
   PyTorchModelHubMixin,  # 继承 mixin
   coders={
      Namespace : (
         lambda x: vars(x),  # 编码器：如何将 Namespace 转换为可用于 JSON 的值？
         lambda data: Namespace(**data),  # 解码器：如何从字典重建 Namespace？
      )
   }
):
    def __init__(self, args: Namespace): # 标注 `args`
      self.pattern = self.args.pattern
      self.hidden_size = self.args.hidden_size
      ...
```

在上面的代码片段中，内部逻辑和类的 `__init__` 签名都没有改变。这意味着库中现有的所有类实例化代码片段都可以继续工作。为此，我们需要：

1. 继承 mixin（这里是 `PytorchModelHubMixin`）。
2. 在继承时传入 `coders` 参数。这是一个字典，键是需要处理的自定义类型，值是一个 `(encoder, decoder)` 元组。
   - 编码器接收指定类型的对象作为输入，并返回可用于 JSON 的值。使用 `save_pretrained` 保存模型时会使用它。
   - 解码器接收原始数据（通常是一个字典）作为输入，并从中重建初始对象。使用 `from_pretrained` 加载模型时会使用它。
3. 为 `__init__` 签名添加类型注解。这一点很重要，因为 mixin 需要知道类所需的类型，从而确定要使用哪个解码器。

为简单起见，上例中的编码器和解码器函数并不健壮。在实际实现中，您很可能需要正确处理各种边界情况。

## 快速比较

下面快速总结前面介绍的两种方案及其优缺点。下表仅供参考。您的框架可能有一些需要处理的特殊情况。本指南只是为集成提供建议和思路。无论如何，如果您有任何问题，欢迎联系我们！

<!-- 使用 https://www.tablesgenerator.com/markdown_tables 生成 -->
|           集成方式           |                                                      使用辅助方法                                                       |                                     使用 [`ModelHubMixin`]                                     |
| :-----------------------------: | :----------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------: |
|         用户体验         |                                `model = load_from_hub(...)`<br>`push_to_hub(model, ...)`                                 |               `model = MyModel.from_pretrained(...)`<br>`model.push_to_hub(...)`                |
|           灵活性           |                                 非常灵活。<br>您可以完全控制实现。                                  |                    灵活性较低。<br>您的框架必须包含模型类。                    |
|           维护成本           | 需要自行维护配置支持和新功能，可能还需要修复用户报告的问题。 | 大多数与 Hub 的交互都由 `huggingface_hub` 实现，因此维护成本较低。 |
| 文档/类型注解 |                                                 需要手动编写。                                                  |                             由 `huggingface_hub` 部分处理。                             |
|         下载计数         |                                                 需要手动处理。                                                  |                      如果类具有 `config` 属性，则默认启用。                      |
|           模型卡片           |                                                  需要手动处理。                                                  |                       默认生成，包含 library_name、tags 等信息。                        |
