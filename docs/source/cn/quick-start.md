<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 快速入门

[Hugging Face Hub](https://huggingface.co/)是分享机器学习模型、演示、数据集和指标的首选平台`huggingface_hub`库帮助你在不离开开发环境的情况下与 Hub 进行交互。你可以轻松地创建和管理仓库,下载和上传文件,并从 Hub 获取有用的模型和数据集元数据

## 安装

要开始使用,请安装`huggingface_hub`库:

```bash
pip install --upgrade huggingface_hub
```

更多详细信息,请查看[安装指南](installation)

## 下载文件

Hugging Face 平台上的存储库是使用 git 版本控制的，用户可以下载单个文件或整个存储库。您可以使用 [`hf_hub_download`] 函数下载文件。该函数将下载并将文件缓存在您的本地磁盘上。下次您需要该文件时，它将从您的缓存中加载，因此您无需重新下载它

您将需要填写存储库 ID 和您要下载的文件的文件名。例如，要下载[Pegasus](https://huggingface.co/google/pegasus-xsum)模型配置文件，请运行以下代码：

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(repo_id="google/pegasus-xsum", filename="config.json")
repo_id: 仓库的 ID 或路径，这里使用了 "google/pegasus-xsum"
filename: 要下载的文件名，这里是 "config.json"
```

要下载文件的特定版本，请使用`revision`参数指定分支名称、标签或提交哈希。如果您选择使用提交哈希，它必须是完整长度的哈希，而不是较短的7个字符的提交哈希：

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(
...     repo_id="google/pegasus-xsum",
...     filename="config.json",
...   revision="4d33b01d79672f27f001f6abade33f22d993b151"
... )
```

有关更多详细信息和选项，请参阅 [`hf_hub_download`] 的 API 参考文档

## 登录

在许多情况下，您必须使用 Hugging Face 帐户进行登录后才能与 Hugging Face 模型库进行交互，例如下载私有存储库、上传文件、创建 PR 等。如果您还没有帐户，请[创建一个](https://huggingface.co/join),然后登录以获取您的 [用户访问令牌](https://huggingface.co/docs/hub/security-tokens),security-tokens从您的[设置页面](https://huggingface.co/settings/tokens)进入设置,用户访问令牌用于向模型库进行身份验证

运行以下代码，这将使用您的用户访问令牌登录到Hugging Face模型库

```bash
hf auth login

hf auth login --token $HUGGINGFACE_TOKEN
```

或者，你可以在笔记本电脑或脚本中使用 [`login`] 来进行程序化登录,请运行以下代码:

```py
>>> from huggingface_hub import login
>>> login()
```

您还可以直接将令牌传递给 [`login`]，如下所示：`login(token="hf_xxx")`。这将使用您的用户访问令牌登录到 Hugging Face 模型库，而无需您输入任何内容。但是，如果您这样做，请在共享源代码时要小心。最好从安全保管库中加载令牌，而不是在代码库/笔记本中显式保存它

您一次只能登录一个帐户。如果您使用另一个帐户登录您的机器，您将会从之前的帐户注销。请确保使用命令 `hf auth whoami`来检查您当前使用的是哪个帐户。如果您想在同一个脚本中处理多个帐户，您可以在调用每个方法时提供您的令牌。这对于您不想在您的机器上存储任何令牌也很有用

> [!WARNING]
> 一旦您登录了，所有对模型库的请求（即使是不需要认证的方法）都将默认使用您的访问令牌。如果您想禁用对令牌的隐式使用，您应该设置`HF_HUB_DISABLE_IMPLICIT_TOKEN`环境变量

## 创建存储库

一旦您注册并登录，请使用 [`create_repo`] 函数创建存储库：

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model")
```
如果您想将存储库设置为私有，请按照以下步骤操作：

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model", private=True)
```
私有存储库将不会对任何人可见，除了您自己

> [!TIP]
> 创建存储库或将内容推送到 Hub 时，必须提供具有`写入`权限的用户访问令牌。您可以在创建令牌时在您的[设置页面](https://huggingface.co/settings/tokens)中选择权限

## 上传文件

您可以使用 [`upload_file`] 函数将文件添加到您新创建的存储库。您需要指定：

1. 要上传的文件的路径

2. 文件在存储库中的位置

3. 您要将文件添加到的存储库的 ID

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.upload_file(
...     path_or_fileobj="/home/lysandre/dummy-test/README.md"
...     path_in_repo="README.md"
...     repo_id="lysandre/test-model"
... )
```

要一次上传多个文件，请查看[上传指南](./guides/upload) ,该指南将向您介绍几种上传文件的方法（有或没有 git）。

## 下一步

`huggingface_hub`库为用户提供了一种使用Python与Hub 进行交互的简单方法。要了解有关如何在Hub上管理文件和存储库的更多信息，我们建议您阅读我们的[操作方法指南](./guides/overview)：

- [管理您的存储库](./guides/repository)
- [从Hub下载文件](./guides/download)
- [将文件上传到Hub](./guides/upload)
- [在Hub中搜索您的所需模型或数据集](./guides/search)
- [了解如何使用 Inference API 进行快速推理](./guides/inference)
