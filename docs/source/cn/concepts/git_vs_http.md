<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Git 与 HTTP 范式

`huggingface_hub`库是用于与Hugging Face Hub进行交互的库，Hugging Face Hub是一组基于Git的存储库（模型、数据集或Spaces）。使用 `huggingface_hub`有两种主要方式来访问Hub。

第一种方法，即所谓的“基于git”的方法，由[`Repository`]类驱动。这种方法使用了一个包装器，它在 `git`命令的基础上增加了专门与Hub交互的额外函数。第二种选择，称为“基于HTTP”的方法，涉及使用[`HfApi`]客户端进行HTTP请求。让我们来看一看每种方法的优缺点。

## 存储库：基于历史的 Git 方法

最初，`huggingface_hub`主要围绕 [`Repository`] 类构建。它为常见的 `git` 命令（如 `"git add"`、`"git commit"`、`"git push"`、`"git tag"`、`"git checkout"` 等）提供了 Python 包装器

该库还可以帮助设置凭据和跟踪大型文件，这些文件通常在机器学习存储库中使用。此外，该库允许您在后台执行其方法，使其在训练期间上传数据很有用。

使用 [`Repository`] 的最大优点是它允许你在本地机器上维护整个存储库的本地副本。这也可能是一个缺点，因为它需要你不断更新和维护这个本地副本。这类似于传统软件开发中，每个开发人员都维护自己的本地副本，并在开发功能时推送更改。但是，在机器学习的上下文中，这可能并不总是必要的，因为用户可能只需要下载推理所需的权重，或将权重从一种格式转换为另一种格式，而无需克隆整个存储库。

## HfApi: 一个功能强大且方便的HTTP客户端

`HfApi` 被开发为本地 git 存储库的替代方案，因为本地 git 存储库在处理大型模型或数据集时可能会很麻烦。`HfApi` 提供与基于 git 的方法相同的功能，例如下载和推送文件以及创建分支和标签，但无需本地文件夹来保持同步。

`HfApi`除了提供 `git` 已经提供的功能外，还提供其他功能，例如：

* 管理存储库
* 使用缓存下载文件以进行有效的重复使用
* 在 Hub 中搜索存储库和元数据
* 访问社区功能，如讨论、PR和评论
* 配置Spaces

## 我应该使用什么？以及何时使用？

总的来说，在大多数情况下，`HTTP 方法`是使用 huggingface_hub 的推荐方法。但是，在以下几种情况下，维护本地 git 克隆（使用 `Repository`）可能更有益：

如果您在本地机器上训练模型，使用传统的 git 工作流程并定期推送更新可能更有效。`Repository` 被优化为此类情况，因为它能够在后台运行。
如果您需要手动编辑大型文件，`git `是最佳选择，因为它只会将文件的差异发送到服务器。使用 `HfAPI` 客户端，每次编辑都会上传整个文件。请记住，大多数大型文件是二进制文件，因此无法从 git 差异中受益。

并非所有 git 命令都通过 [`HfApi`] 提供。有些可能永远不会被实现，但我们一直在努力改进并缩小差距。如果您没有看到您的用例被覆盖。

请在[Github](https://github.com/huggingface/huggingface_hub)打开一个 issue！我们欢迎反馈，以帮助我们与我们的用户一起构建 🤗 生态系统。
