<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Quickstart  快速入门

The [Hugging Face Hub](https://huggingface.co/) is the go-to place for sharing machine learning
models, demos, datasets, and metrics. `huggingface_hub` library helps you interact with
the Hub without leaving your development environment. You can create and manage
repositories easily, download and upload files, and get useful model and dataset
metadata from the Hub.

Hugging Face Hub 是分享机器学习模型、演示、数据集和指标的首选平台。huggingface_hub 库帮助你在不离开开发环境的情况下与 Hub 进行交互。你可以轻松地创建和管理仓库,下载和上传文件,并从 Hub 获取有用的模型和数据集元数据。

## Installation   安装

To get started, install the `huggingface_hub` library:

要开始使用,请安装 huggingface_hub 库:

```bash
pip install --upgrade huggingface_hub    ## 使用pip安装huggingface_hub库
```

For more details, check out the [installation](installation) guide.

更多详细信息,请查看安装指南

## Download files   下载文件

Repositories on the Hub are git version controlled, and users can download a single file
or the whole repository. You can use the [`hf_hub_download`] function to download files.
This function will download and cache a file on your local disk. The next time you need
that file, it will load from your cache, so you don't need to re-download it.

Hugging Face 平台上的存储库是使用 git 版本控制的，用户可以下载单个文件或整个存储库。您可以使用 [hf_hub_download] 函数下载文件。该函数将下载并将文件缓存在您的本地磁盘上。下次您需要该文件时，它将从您的缓存中加载，因此您无需重新下载它

You will need the repository id and the filename of the file you want to download. For
example, to download the [Pegasus](https://huggingface.co/google/pegasus-xsum) model
configuration file: 

您将需要填写存储库 ID 和您要下载的文件的文件名。例如，要下载
Pegasus:https://huggingface.co/google/pegasus-xsum 模型配置文件：

```py
>>> from huggingface_hub import hf_hub_download   # 导入了从 huggingface_hub 模块中的 hf_hub_download 函数
>>> hf_hub_download(repo_id="google/pegasus-xsum", filename="config.json")   #使用 hf_hub_download 函数下载特定仓库中的文件
repo_id: 仓库的 ID 或路径，这里使用了 "google/pegasus-xsum"
filename: 要下载的文件名，这里是 "config.json"
```

To download a specific version of the file, use the `revision` parameter to specify the
branch name, tag, or commit hash. If you choose to use the commit hash, it must be the
full-length hash instead of the shorter 7-character commit hash: 

要下载文件的特定版本，请使用 revision 参数指定分支名称、标签或提交哈希。如果您选择使用提交哈希，它必须是完整长度的哈希，而不是较短的 7 个字符的提交哈希：

```py
>>> from huggingface_hub import hf_hub_download  ## 从 huggingface_hub 模块中导入 hf_hub_download 函数
>>> hf_hub_download(
...     repo_id="google/pegasus-xsum",   # 下载来自于 "google/pegasus-xsum" 仓库
...     filename="config.json",   # 下载的文件名是 "config.json"
...   revision="4d33b01d79672f27f001f6abade33f22d993b151"  # 使用特定的版本或提交哈希值进行下载
... )
```
For more details and options, see the API reference for [`hf_hub_download`].

有关更多详细信息和选项，请参阅 [hf_hub_download] 的 API 参考文档

## Login   登录

In a lot of cases, you must be logged in with a Hugging Face account to interact with
the Hub: download private repos, upload files, create PRs,...
[Create an account](https://huggingface.co/join) if you don't already have one, and then sign in
to get your [User Access Token](https://huggingface.co/docs/hub/security-tokens) from
your [Settings page](https://huggingface.co/settings/tokens). The User Access Token is
used to authenticate your identity to the Hub.

在许多情况下，您必须使用 Hugging Face 帐户进行登录后才能与 Hugging Face 模型库进行交互，例如下载私有存储库、上传文件、创建 PR 等。如果您还没有帐户，请创建一个: https://huggingface.co/join，然后登录以获取您的用户访问令牌: https://huggingface.co/docs/hub/security-tokens从您的设置页面: https://huggingface.co/settings/tokens。用户访问令牌用于向模型库进行身份验证。

Once you have your User Access Token, run the following command in your terminal:

这将使用您的用户访问令牌登录到 Hugging Face 模型库

```bash
huggingface-cli login       # 使用 huggingface-cli 登录命令
# or using an environment variable
huggingface-cli login --token $HUGGINGFACE_TOKEN    # 或者可以使用环境变量进行登录
```

Alternatively, you can programmatically login using [`login`] in a notebook or a script:

或者，你可以在笔记本电脑或脚本中使用 [login] 来进行程序化登录

```py
>>> from huggingface_hub import login   # 从 huggingface_hub 模块中导入 login 函数
>>> login()    # 使用 login 函数进行登录
```

It is also possible to login programmatically without being prompted to enter your token by directly
passing the token to [`login`] like `login(token="hf_xxx")`. If you do so, be careful when
sharing your source code. It is a best practice to load the token from a secure vault instead
of saving it explicitly in your codebase/notebook.

您还可以直接将令牌传递给 [login]，如下所示：login(token="hf_xxx")这将使用您的用户访问令牌登录到 Hugging Face 模型库，而无需您输入任何内容。但是，如果您这样做，请在共享源代码时要小心。最好从安全保管库中加载令牌，而不是在代码库/笔记本中显式保存它

You can be logged in only to 1 account at a time. If you login your machine to a new account, you will get logged out
from the previous. Make sure to always which account you are using with the command `huggingface-cli whoami`.
If you want to handle several accounts in the same script, you can provide your token when calling each method. This
is also useful if you don't want to store any token on your machine.

您一次只能登录一个帐户。如果您使用另一个帐户登录您的机器，您将会从之前的帐户注销。请确保使用命令 huggingface-cli whoami 来检查您当前使用的是哪个帐户。如果您想在同一个脚本中处理多个帐户，您可以在调用每个方法时提供您的令牌。这对于您不想在您的机器上存储任何令牌也很有用。

<Tip warning={true}>

Once you are logged in, all requests to the Hub -even methods that don't necessarily require authentication- will use your
access token by default. If you want to disable implicit use of your token, you should set the
`HF_HUB_DISABLE_IMPLICIT_TOKEN` environment variable.

一旦您登录了，所有对模型库的请求（即使是不需要认证的方法）都将默认使用您的访问令牌。如果您想禁用对令牌的隐式使用，您应该设置 HF_HUB_DISABLE_IMPLICIT_TOKEN 环境变量。

</Tip>

## Create a repository    创建存储库
 
Once you've registered and logged in, create a repository with the [`create_repo`]
function:

一旦您注册并登录，请使用 [`create_repo`] 函数创建存储库：

```py
>>> from huggingface_hub import HfApi  # 从 huggingface_hub 模块中导入 HfApi 函数
>>> api = HfApi()   # 创建一个 HfApi 实例
>>> api.create_repo(repo_id="super-cool-model")   # 使用 create_repo 方法创建一个新的仓库，repo_id: 要创建的仓库的 ID 或路径，这里是 "super-cool-model"
```

If you want your repository to be private, then:

如果您想将存储库设置为私有，请按照以下步骤操作：

```py
>>> from huggingface_hub import HfApi  # 从 huggingface_hub 模块中导入 HfApi 函数
>>> api = HfApi()  # 创建一个 HfApi 实例
>>> api.create_repo(repo_id="super-cool-model", private=True)  # 使用 create_repo 方法创建一个新的私有仓库，repo_id: 要创建的仓库的 ID 或路径，这里是 "super-cool-model"， private: 指定是否创建私有仓库，这里设置为 True，表示创建私有仓库
```

Private repositories will not be visible to anyone except yourself.

私有存储库将不会对任何人可见，除了您自己

<Tip>

To create a repository or to push content to the Hub, you must provide a User Access
Token that has the `write` permission. You can choose the permission when creating the
token in your [Settings page](https://huggingface.co/settings/tokens).

创建存储库或将内容推送到 Hub 时，必须提供具有 写入 权限的用户访问令牌。您可以在创建令牌时在您的设置页面: https://huggingface.co/settings/tokens 中选择权限

</Tip>

## Upload files   上传文件

Use the [`upload_file`] function to add a file to your newly created repository. You
need to specify:

您可以使用 [upload_file()] 函数将文件添加到您新创建的存储库。您需要指定：

1. The path of the file to upload.

要上传的文件的路径
   
2. The path of the file in the repository.

文件在存储库中的位置
   
3. The repository id of where you want to add the file.

您要将文件添加到的存储库的 ID

```py
>>> from huggingface_hub import HfApi  # 从 huggingface_hub 模块中导入 HfApi 函数
>>> api = HfApi()  # 创建一个 HfApi 实例
>>> api.upload_file(         
...     path_or_fileobj="/home/lysandre/dummy-test/README.md"   # path_or_fileobj: 要上传的文件的路径或文件对象，这里是 "/home/lysandre/dummy-test/README.md,  
...     path_in_repo="README.md"  # path_in_repo: 文件在仓库中的路径，这里是 "README.md" ,
...     repo_id="lysandre/test-model"  # repo_id: 目标仓库的 ID 或路径，这里是 "lysandre/test-model" ,
... )
```

To upload more than one file at a time, take a look at the [Upload](./guides/upload) guide
which will introduce you to several methods for uploading files (with or without git).

要一次上传多个文件，请查看上传: ./guides/upload指南，该指南将向您介绍几种上传文件的方法（有或没有 git）。

## Next steps   下一步

The `huggingface_hub` library provides an easy way for users to interact with the Hub
with Python. To learn more about how you can manage your files and repositories on the
Hub, we recommend reading our [how-to guides](./guides/overview) to:

huggingface_hub 库为用户提供了一种使用 Python 与 Hub 进行交互的简单方法。要了解有关如何在 Hub 上管理文件和存储库的更多信息，我们建议您阅读我们的 操作方法指南: ./guides/overview：

- [Manage your repository](./guides/repository).  管理您的存储库
- [Download](./guides/download) files from the Hub.  从 Hub 下载文件
- [Upload](./guides/upload) files to the Hub.  将文件上传到 Hub
- [Search the Hub](./guides/search) for your desired model or dataset.  在 Hub 中搜索您的所需模型或数据集
- [Access the Inference API](./guides/inference) for fast inference.  了解如何使用 Inference API 进行快速推理。
