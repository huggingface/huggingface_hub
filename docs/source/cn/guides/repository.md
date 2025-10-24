<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 创建和管理存储库

Hugging Face Hub是一组 Git 存储库。[Git](https://git-scm.com/)是软件开发中广泛使用的工具，可以在协作工作时轻松对项目进行版本控制。本指南将向您展示如何与 Hub 上的存储库进行交互，特别关注以下内容：

- 创建和删除存储库
- 管理分支和标签
- 重命名您的存储库
- 更新您的存储库可见性
- 管理存储库的本地副本

> [!WARNING]
> 如果您习惯于使用类似于GitLab/GitHub/Bitbucket等平台，您可能首先想到使用 `git`命令行工具来克隆存储库（`git clone`）、提交更改（`git add` , ` git commit`）并推送它们（`git push`）。在使用 Hugging Face Hub 时，这是有效的。然而，软件工程和机器学习并不具有相同的要求和工作流程。模型存储库可能会维护大量模型权重文件以适应不同的框架和工具，因此克隆存储库会导致您维护大量占用空间的本地文件夹。因此，使用我们的自定义HTTP方法可能更有效。您可以阅读我们的[git与HTTP相比较](../concepts/git_vs_http)解释页面以获取更多详细信息

如果你想在Hub上创建和管理一个仓库，你的计算机必须处于登录状态。如果尚未登录，请参考[此部分](../quick-start#login)。在本指南的其余部分，我们将假设你的计算机已登录

## 仓库创建和删除

第一步是了解如何创建和删除仓库。你只能管理你拥有的仓库（在你的用户名命名空间下）或者你具有写入权限的组织中的仓库

### 创建一个仓库

使用 [`create_repo`] 创建一个空仓库，并通过 `repo_id`参数为其命名 `repo_id`是你的命名空间，后面跟着仓库名称：`username_or_org/repo_name`

运行以下代码，以创建仓库：

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-model")
'https://huggingface.co/lysandre/test-model'
```

默认情况下，[`create_repo`] 会创建一个模型仓库。但是你可以使用 `repo_type`参数来指定其他仓库类型。例如，如果你想创建一个数据集仓库

请运行以下代码：

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-dataset", repo_type="dataset")
'https://huggingface.co/datasets/lysandre/test-dataset'
```

创建仓库时，你可以使用 `private`参数设置仓库的可见性

请运行以下代码

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-private", private=True)
```

如果你想在以后更改仓库的可见性，你可以使用[`update_repo_settings`] 函数

### 删除一个仓库

使用 [`delete_repo`] 删除一个仓库。确保你确实想要删除仓库，因为这是一个不可逆转的过程！做完上述过程后，指定你想要删除的仓库的 `repo_id`

请运行以下代码：

```py
>>> delete_repo(repo_id="lysandre/my-corrupted-dataset", repo_type="dataset")
```

### 克隆一个仓库（仅适用于 Spaces）

在某些情况下，你可能想要复制别人的仓库并根据自己的用例进行调整。对于 Spaces，你可以使用 [`duplicate_space`] 方法来实现。它将复制整个仓库。

你仍然需要配置自己的设置（硬件和密钥）。查看我们的[管理你的Space指南](./manage-spaces)以获取更多详细信息。

请运行以下代码：

```py
>>> from huggingface_hub import duplicate_space
>>> duplicate_space("multimodalart/dreambooth-training", private=False)
RepoUrl('https://huggingface.co/spaces/nateraw/dreambooth-training',...)
```

## 上传和下载文件

既然您已经创建了您的存储库，您现在也可以推送更改至其中并从中下载文件

这两个主题有它们自己的指南。请[上传指南](./upload) 和[下载指南](./download)来学习如何使用您的存储库。

## 分支和标签

Git存储库通常使用分支来存储同一存储库的不同版本。标签也可以用于标记存储库的特定状态，例如，在发布版本这个情况下。更一般地说，分支和标签被称为[git引用](https://git-scm.com/book/en/v2/Git-Internals-Git-References).

### 创建分支和标签

你可以使用[`create_branch`]和[`create_tag`]来创建新的分支和标签:

请运行以下代码：

```py
>>> from huggingface_hub import create_branch, create_tag

# Create a branch on a Space repo from `main` branch
>>> create_branch("Matthijs/speecht5-tts-demo", repo_type="space", branch="handle-dog-speaker")

# Create a tag on a Dataset repo from `v0.1-release` branch
>>> create_branch("bigcode/the-stack", repo_type="dataset", revision="v0.1-release", tag="v0.1.1", tag_message="Bump release version.")
```

同时,你可以以相同的方式使用 [`delete_branch`] 和 [`delete_tag`] 函数来删除分支或标签

### 列出所有的分支和标签

你还可以使用 [`list_repo_refs`] 列出存储库中的现有 Git 引用
请运行以下代码：

```py
>>> from huggingface_hub import list_repo_refs
>>> api.list_repo_refs("bigcode/the-stack", repo_type="dataset")
GitRefs(
   branches=[
         GitRefInfo(name='main', ref='refs/heads/main', target_commit='18edc1591d9ce72aa82f56c4431b3c969b210ae3'),
         GitRefInfo(name='v1.1.a1', ref='refs/heads/v1.1.a1', target_commit='f9826b862d1567f3822d3d25649b0d6d22ace714')
   ],
   converts=[],
   tags=[
         GitRefInfo(name='v1.0', ref='refs/tags/v1.0', target_commit='c37a8cd1e382064d8aced5e05543c5f7753834da')
   ]
)
```

## 修改存储库设置

存储库具有一些可配置的设置。大多数情况下，您通常会在浏览器中的存储库设置页面上手动配置这些设置。要配置存储库，您必须具有对其的写访问权限（拥有它或属于组织）。在本节中，我们将看到您还可以使用 `huggingface_hub` 在编程方式上配置的设置。

一些设置是特定于 Spaces（硬件、环境变量等）的。要配置这些设置，请参考我们的[管理Spaces](../guides/manage-spaces)指南。

### 更新可见性

一个存储库可以是公共的或私有的。私有存储库仅对您或存储库所在组织的成员可见。

请运行以下代码将存储库更改为私有：

```py
>>> from huggingface_hub import update_repo_settings
>>> update_repo_settings(repo_id=repo_id, private=True)
```

### 重命名您的存储库

您可以使用 [`move_repo`] 在 Hub 上重命名您的存储库。使用这种方法，您还可以将存储库从一个用户移动到一个组织。在这样做时，有一些[限制](https://hf.co/docs/hub/repositories-settings#renaming-or-transferring-a-repo)需要注意。例如，您不能将存储库转移到另一个用户。

请运行以下代码：

```py
>>> from huggingface_hub import move_repo
>>> move_repo(from_id="Wauplin/cool-model", to_id="huggingface/cool-model")
```
