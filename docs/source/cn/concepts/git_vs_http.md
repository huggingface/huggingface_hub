<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Git 与 HTTP 范式

`huggingface_hub` 是用于与 Hugging Face Hub 交互的库。Hub 是一组基于 git 的仓库（模型、数据集或 Spaces）。使用 `huggingface_hub` 访问 Hub 主要有两种方式。

第一种是所谓的“基于 git”的方式，直接在终端中使用标准 `git` 命令。这种方法可以克隆仓库、创建提交，并手动推送更改。第二种是“基于 HTTP”的方式，通过 [`HfApi`] 客户端发出 HTTP 请求。下面我们来比较这两种方式的优缺点。

## Git：历史上基于 CLI 的方式

起初，大多数用户使用普通的 `git` 命令与 Hugging Face Hub 交互，例如 `git clone`、`git add`、`git commit`、`git push`、`git tag` 或 `git checkout`。

这种方式可以在本地保留仓库的完整副本，就像传统软件开发一样。当您需要离线访问，或希望使用仓库的完整历史时，这会是一个优势。但它也有缺点：您需要自己负责在本地保持仓库最新、处理凭据，以及通过 `git-lfs` 管理大文件。在处理大型机器学习模型或数据集时，这些工作会变得很繁琐。

在许多机器学习工作流中，您可能只需要下载少量文件用于推理，或转换权重，而不需要克隆整个仓库。这种情况下，使用 `git` 会显得过度，并引入不必要的复杂性。

## HfApi：灵活便捷的 HTTP 客户端

[`HfApi`] 类的出现，是为了替代需要在本地维护 git 仓库的做法——尤其是在处理大型模型或数据集时，本地仓库会很难维护。[`HfApi`] 提供了与基于 git 的工作流相同的功能，例如下载和推送文件、创建分支和标签，但不再需要一个必须保持同步的本地文件夹。

除了 `git` 已有的功能外，[`HfApi`] 还提供了更多能力，例如管理仓库、通过缓存高效复用已下载的文件、在 Hub 上搜索仓库和元数据、使用讨论、PR 和评论等社区功能，以及配置 Spaces 的硬件和密钥。

## 我该用哪种？什么时候用？

总体来说，**在所有场景下都推荐使用基于 HTTP 的方式**。[`HfApi`] 可以拉取和推送更改，处理 PR、标签和分支，参与讨论，以及完成更多操作。

不过，并非所有 git 命令都已通过 [`HfApi`] 提供。有些命令可能永远不会实现，但我们一直在改进并缩小差距。如果您发现自己的用例尚未覆盖，请在 GitHub 上[提交 issue](https://github.com/huggingface/huggingface_hub)！我们欢迎反馈，以便与用户一起构建 HF 生态系统。

优先使用基于 HTTP 的 [`HfApi`]、而不是直接使用 `git` 命令，并不意味着 git 版本管理会很快从 Hugging Face Hub 消失。在适合使用 `git` 的工作流中，您始终可以在本地使用它。
