<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub.svg">
    <img alt="huggingface_hub library logo" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p> 

<p align="center">
    <i>Hugging Face Hub Python 客户端</i>
</p>

<p align="center">
    <a href="https://huggingface.co/docs/huggingface_hub/en/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/huggingface_hub/index.svg?down_color=red&down_message=offline&up_message=online&label=doc"></a>
    <a href="https://github.com/huggingface/huggingface_hub/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/huggingface_hub.svg"></a>
    <a href="https://github.com/huggingface/huggingface_hub"><img alt="PyPi version" src="https://img.shields.io/pypi/pyversions/huggingface_hub.svg"></a>
    <a href="https://pypi.org/project/huggingface-hub"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/huggingface_hub"></a>
    <a href="https://codecov.io/gh/huggingface/huggingface_hub"><img alt="Code coverage" src="https://codecov.io/gh/huggingface/huggingface_hub/branch/main/graph/badge.svg?token=RXP95LE2XL"></a>
</p>

<h4 align="center">
    <p>
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/README.md">English</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_hi.md">हिंदी</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_kn.md">ಕನ್ನಡ</a> |
        <b>中文（简体）</b>
    <p>
</h4>

---

**文档**: <a href="https://hf.co/docs/huggingface_hub" target="_blank">https://hf.co/docs/huggingface_hub </a>

**源代码**: <a href="https://github.com/huggingface/huggingface_hub" target="_blank">https://github.com/huggingface/huggingface_hub </a>

---

## 欢迎使用 Hugging Face Hub 库

通过`huggingface_hub` 库，您可以与面向机器学习开发者和协作者的平台 [Hugging Face Hub](https://huggingface.co/)进行交互，找到适用于您所在项目的预训练模型和数据集，体验在平台托管的数百个机器学习应用，还可以创建或分享自己的模型和数据集并于社区共享。以上所有都可以用Python在`huggingface_hub` 库中轻松实现。

## 主要特点

- [从hugging face hub下载文件](https://huggingface.co/docs/huggingface_hub/en/guides/download)
- [上传文件到 hugging face hub](https://huggingface.co/docs/huggingface_hub/en/guides/upload)
- [管理您的存储库](https://huggingface.co/docs/huggingface_hub/en/guides/repository)
- [在部署的模型上运行推断](https://huggingface.co/docs/huggingface_hub/en/guides/inference)
- [搜索模型、数据集和空间](https://huggingface.co/docs/huggingface_hub/en/guides/search)
- [分享模型卡片](https://huggingface.co/docs/huggingface_hub/en/guides/model-cards)
- [社区互动](https://huggingface.co/docs/huggingface_hub/en/guides/community)

## 安装

使用pip安装 `huggingface_hub` 包：

```bash
pip install huggingface_hub
```

如果您更喜欢，也可以使用 conda 进行安装

为了默认保持包的最小化，huggingface_hub 带有一些可选的依赖项，适用于某些用例。例如，如果您想要完整的推断体验，请运行：

```bash
pip install huggingface_hub[inference]
```

要了解更多安装和可选依赖项，请查看[安装指南](https://huggingface.co/docs/huggingface_hub/cn/安装)

## 快速入门指南

### 下载文件

下载单个文件,请运行以下代码：

```py
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="tiiuae/falcon-7b-instruct", filename="config.json")
```

如果下载整个存储库，请运行以下代码：

```py
from huggingface_hub import snapshot_download

snapshot_download("stabilityai/stable-diffusion-2-1")
```

文件将被下载到本地缓存文件夹。更多详细信息请参阅此 [指南](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache).

### 登录

Hugging Face Hub 使用令牌对应用进行身份验证(请参阅[文档](https://huggingface.co/docs/hub/security-tokens)). 要登录您的机器，请运行以下命令行：

```bash
hf auth login
# or using an environment variable
hf auth login --token $HUGGINGFACE_TOKEN
```

### 创建一个存储库

要创建一个新存储库，请运行以下代码：

```py
from huggingface_hub import create_repo

create_repo(repo_id="super-cool-model")
```

### 上传文件

上传单个文件,请运行以下代码

```py
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="/home/lysandre/dummy-test/README.md",
    path_in_repo="README.md",
    repo_id="lysandre/test-model",
)
```

如果上传整个存储库，请运行以下代码：

```py
from huggingface_hub import upload_folder

upload_folder(
    folder_path="/path/to/local/space",
    repo_id="username/my-cool-space",
    repo_type="space",
)
```

有关详细信息，请查看 [上传指南](https://huggingface.co/docs/huggingface_hub/en/guides/upload).

## 集成到 Hub 中

我们正在与一些出色的开源机器学习库合作，提供免费的模型托管和版本控制。您可以在 [这里](https://huggingface.co/docs/hub/libraries)找到现有的集成

优势包括:

- 为库及其用户提供免费的模型或数据集托管
- 内置文件版本控制，即使对于非常大的文件也能实现，这得益于基于 Git 的方法
- 为所有公开可用的模型提供托管的推断 API
- 在网页端可在线体验所有公开的模型
- 任何人都可以上传新模型到您的库，他们只需为模型添加相应的标签，以便让其被发现
- 快速下载！我们使用 Cloudfront（CDN）进行地理复制下载，因此无论在全球任何地方，下载速度都非常快。
- 使用统计和更多功能即将推出

如果您想要集成您的库，请随时打开一个问题来开始讨论。我们编写了一份逐步指南，以❤️的方式展示如何进行这种集成。

## 欢迎各种贡献（功能请求、错误等） 💙💚💛💜🧡❤️

欢迎每个人来进行贡献，我们重视每个人的贡献。编写代码并非唯一的帮助社区的方式。回答问题、帮助他人、积极互动并改善文档对社区来说都是极其有价值的。为此我们编写了一份 [贡献指南](https://github.com/huggingface/huggingface_hub/blob/main/CONTRIBUTING.md) 以进行总结，即如何开始为这个存储库做贡献
