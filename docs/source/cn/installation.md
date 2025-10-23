<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 安装

在开始之前，您需要通过安装适当的软件包来设置您的环境

huggingface_hub 在 Python 3.9 或更高版本上进行了测试，可以保证在这些版本上正常运行。如果您使用的是 Python 3.7 或更低版本，可能会出现兼容性问题

## 使用 pip 安装

我们建议将huggingface_hub安装在[虚拟环境](https://docs.python.org/3/library/venv.html)中.
如果你不熟悉 Python虚拟环境,可以看看这个[指南](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

虚拟环境可以更容易地管理不同的项目,避免依赖项之间的兼容性问题

首先在你的项目目录中创建一个虚拟环境,请运行以下代码:

```bash
python -m venv .env
```

在Linux和macOS上,请运行以下代码激活虚拟环境:

```bash
source .env/bin/activate
```

在 Windows 上，请运行以下代码激活虚拟环境:

```bash
.env/Scripts/activate
```

现在您可以从[PyPi注册表](https://pypi.org/project/huggingface-hub/)安装 `huggingface_hub`：

```bash
pip install --upgrade huggingface_hub
```

完成后,[检查安装](#check-installation)是否正常工作

### 安装可选依赖项

`huggingface_hub`的某些依赖项是 [可选](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies) 的，因为它们不是运行`huggingface_hub`的核心功能所必需的.但是，如果没有安装可选依赖项， `huggingface_hub` 的某些功能可能会无法使用

您可以通过`pip`安装可选依赖项,请运行以下代码：

```bash
# 安装 Torch 特定功能和 CLI 特定功能的依赖项
pip install 'huggingface_hub[cli,torch]'
```

这里列出了 `huggingface_hub` 的可选依赖项：

- `cli`：为 `huggingface_hub` 提供更方便的命令行界面

- `fastai`,` torch`: 运行框架特定功能所需的依赖项

- `dev`：用于为库做贡献的依赖项。包括 `testing`（用于运行测试）、`typing`（用于运行类型检查器）和 `quality`（用于运行 linter）

### 从源代码安装

在某些情况下，直接从源代码安装`huggingface_hub`会更有趣。因为您可以使用最新的主版本`main`而非最新的稳定版本

`main`版本更有利于跟进平台的最新开发进度，例如，在最近一次官方发布之后和最新的官方发布之前所修复的某个错误

但是，这意味着`main`版本可能不总是稳定的。我们会尽力让其正常运行，大多数问题通常会在几小时或一天内解决。如果您遇到问题，请创建一个 Issue ，以便我们可以更快地解决！

```bash
pip install git+https://github.com/huggingface/huggingface_hub  # 使用pip从GitHub仓库安装Hugging Face Hub库
```

从源代码安装时，您还可以指定特定的分支。如果您想测试尚未合并的新功能或新错误修复，这很有用

```bash
pip install git+https://github.com/huggingface/huggingface_hub@my-feature-branch  # 使用pip从指定的GitHub分支（my-feature-branch）安装Hugging Face Hub库
```

完成安装后，请[检查安装](#check-installation)是否正常工作

### 可编辑安装

从源代码安装允许您设置[可编辑安装](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs).如果您计划为`huggingface_hub`做出贡献并需要测试代码更改，这是一个更高级的安装方式。您需要在本机上克隆一个`huggingface_hub`的本地副本

```bash
# 第一，使用以下命令克隆代码库
git clone https://github.com/huggingface/huggingface_hub.git

# 然后，使用以下命令启动虚拟环境
cd huggingface_hub
pip install -e .
```

这些命令将你克隆存储库的文件夹与你的 Python 库路径链接起来。Python 现在将除了正常的库路径之外，还会在你克隆到的文件夹中查找。例如，如果你的 Python 包通常安装在`./.venv/lib/python3.13/site-packages/`中，Python 还会搜索你克隆的文件夹`./huggingface_hub/`

## 通过 conda 安装

如果你更熟悉它，你可以使用[conda-forge channel](https://anaconda.org/conda-forge/huggingface_hub)渠道来安装 `huggingface_hub`

请运行以下代码：

```bash
conda install -c conda-forge huggingface_hub
```
完成安装后，请[检查安装](#check-installation)是否正常工作

## 验证安装

安装完成后，通过运行以下命令检查`huggingface_hub`是否正常工作:

```bash
python -c "from huggingface_hub import model_info; print(model_info('gpt2'))"
```

这个命令将从 Hub 获取有关 [gpt2](https://huggingface.co/gpt2) 模型的信息。

输出应如下所示：

```text
Model Name: gpt2  模型名称
Tags: ['pytorch', 'tf', 'jax', 'tflite', 'rust', 'safetensors', 'gpt2', 'text-generation', 'en', 'doi:10.57967/hf/0039', 'transformers', 'exbert', 'license:mit', 'has_space']  标签
Task: text-generation  任务：文本生成
```

## Windows局限性

为了实现让每个人都能使用机器学习的目标，我们构建了 `huggingface_hub`库，使其成为一个跨平台的库，尤其可以在 Unix 和 Windows 系统上正常工作。但是，在某些情况下，`huggingface_hub`在Windows上运行时会有一些限制。以下是一些已知问题的完整列表。如果您遇到任何未记录的问题，请打开 [Github上的issue](https://github.com/huggingface/huggingface_hub/issues/new/choose).让我们知道

- `huggingface_hub`的缓存系统依赖于符号链接来高效地缓存从Hub下载的文件。在Windows上，您必须激活开发者模式或以管理员身份运行您的脚本才能启用符号链接。如果它们没有被激活，缓存系统仍然可以工作，但效率较低。有关更多详细信息，请阅读[缓存限制](./guides/manage-cache#limitations)部分。

- Hub上的文件路径可能包含特殊字符（例如:`path/to?/my/file`）。Windows对[特殊字符](https://learn.microsoft.com/en-us/windows/win32/intl/character-sets-used-in-file-names)更加严格，这使得在Windows上下载这些文件变得不可能。希望这是罕见的情况。如果您认为这是一个错误，请联系存储库所有者或我们，以找出解决方案。


## 后记

一旦您在机器上正确安装了`huggingface_hub`，您可能需要[配置环境变量](package_reference/environment_variables)或者[查看我们的指南之一](guides/overview)以开始使用。
