<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Installation  安装

Before you start, you will need to setup your environment by installing the appropriate packages.

在开始之前，您需要通过安装适当的软件包来设置您的环境

`huggingface_hub` is tested on **Python 3.8+**.

huggingface_hub 在 Python 3.8 或更高版本上进行了测试，可以保证在这些版本上正常运行。如果您使用的是 Python 3.7 或更低版本，可能会出现兼容性问题

## Install with pip   使用 pip 安装

It is highly recommended to install `huggingface_hub` in a [virtual environment](https://docs.python.org/3/library/venv.html).
If you are unfamiliar with Python virtual environments, take a look at this [guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).
A virtual environment makes it easier to manage different projects, and avoid compatibility issues between dependencies.

建议将huggingface_hub安装在虚拟环境中。如果你不熟悉 Python 虚拟环境,可以看看这个指南。虚拟环境可以更容易地管理不同的项目,避免依赖项之间的兼容性问题。

Start by creating a virtual environment in your project directory:

首先在你的项目目录中创建一个虚拟环境:

```bash
python -m venv .env    # 创建一个名为.env的虚拟环境
```

Activate the virtual environment. On Linux and macOS:

在Linux和macOS上,请运行以下代码激活虚拟环境:

```bash
source .env/bin/activate  # 在Linux和macOS上激活名为.env的虚拟环境
```

Activate virtual environment on Windows:

在 Windows 上，请运行以下代码激活虚拟环境:

```bash
.env/Scripts/activate   #在Windows系统上激活名为.env的虚拟环境
```

Now you're ready to install `huggingface_hub` [from the PyPi registry](https://pypi.org/project/huggingface-hub/):

现在您可以从 PyPi 注册表 https://pypi.org/project/huggingface-hub/: https://pypi.org/project/huggingface-hub/ 安装 huggingface_hub：

```bash
pip install --upgrade huggingface_hub  # 使用pip升级Hugging Face Hub库
```

Once done, [check installation](#check-installation) is working correctly.

完成后，检查安装是否正常工作

### Install optional dependencies  安装可选依赖项

Some dependencies of `huggingface_hub` are [optional](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies) because they are not required to run the core features of `huggingface_hub`. However, some features of the `huggingface_hub` may not be available if the optional dependencies aren't installed.

huggingface_hub 的某些依赖是可选的，因为它们不是运行 huggingface_hub 的核心功能所必需的。但是，如果没有安装可选依赖项，则 huggingface_hub 的某些功能可能无法使用

You can install optional dependencies via `pip`:

您可以通过 pip 安装可选依赖项

```bash
# Install dependencies for tensorflow-specific features  安装 TensorFlow 特定功能的依赖项
# /!\ Warning: this is not equivalent to `pip install tensorflow`  注意：这不等同于 `pip install tensorflow`
pip install 'huggingface_hub[tensorflow]'  # 使用pip安装Hugging Face Hub库，并包括TensorFlow支持的可选依赖项

# Install dependencies for both torch-specific and CLI-specific features.安装 TensorFlow 特定功能和 CLI 特定功能的依赖项
pip install 'huggingface_hub[cli,torch]'    # 使用pip安装Hugging Face Hub库，包括CLI工具和PyTorch支持的可选依赖项
```

Here is the list of optional dependencies in `huggingface_hub`:

这里列出了 `huggingface_hub` 的可选依赖项：

- `cli`: provide a more convenient CLI interface for `huggingface_hub`.

cli：为 huggingface_hub 提供更方便的命令行界面。

- `fastai`, `torch`, `tensorflow`: dependencies to run framework-specific features.

fastai, torch, tensorflow: 运行框架特定功能所需的依赖项

- `dev`: dependencies to contribute to the lib. Includes `testing` (to run tests), `typing` (to run type checker) and `quality` (to run linters).

dev：用于为库做贡献的依赖项。包括 testing（用于运行测试）、typing（用于运行类型检查器）和 quality（用于运行 linter）。

### Install from source  从源代码安装

In some cases, it is interesting to install `huggingface_hub` directly from source.
This allows you to use the bleeding edge `main` version rather than the latest stable version.
The `main` version is useful for staying up-to-date with the latest developments, for instance
if a bug has been fixed since the last official release but a new release hasn't been rolled out yet.

在某些情况下，直接从源代码安装 huggingface_hub 很有趣。这允许您使用最新的稳定版本 main 版本而不是最新的稳定版本。
main 版本对于保持最新的开发进度很有用，例如，如果自上次官方发布以来修复了一个错误，但尚未发布新的版本

However, this means the `main` version may not always be stable. We strive to keep the
`main` version operational, and most issues are usually resolved
within a few hours or a day. If you run into a problem, please open an Issue so we can
fix it even sooner!

但是，这意味着 main 版本可能不总是稳定的。我们会尽力让 main 版本保持可用，并且大多数问题通常会在几小时或一天内解决。如果您遇到问题，请创建一个 Issue ，以便我们可以更快地解决！

```bash
pip install git+https://github.com/huggingface/huggingface_hub  # 使用pip从GitHub仓库安装Hugging Face Hub库
```

When installing from source, you can also specify a specific branch. This is useful if you
want to test a new feature or a new bug-fix that has not been merged yet:

从源代码安装时，您还可以指定特定的分支。如果您想测试尚未合并的新功能或新错误修复，这很有用

```bash
pip install git+https://github.com/huggingface/huggingface_hub@my-feature-branch  # 使用pip从指定的GitHub分支（my-feature-branch）安装Hugging Face Hub库
```

Once done, [check installation](#check-installation) is working correctly.

完成安装后，请[检查安装](#check-installation)是否正常工作。

### Editable install  

Installing from source allows you to setup an [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs).
This is a more advanced installation if you plan to contribute to `huggingface_hub`
and need to test changes in the code. You need to clone a local copy of `huggingface_hub`
on your machine.

```bash
# First, clone repo locally
git clone https://github.com/huggingface/huggingface_hub.git

# Then, install with -e flag
cd huggingface_hub
pip install -e .
```

These commands will link the folder you cloned the repository to and your Python library paths.
Python will now look inside the folder you cloned to in addition to the normal library paths.
For example, if your Python packages are typically installed in `./.venv/lib/python3.11/site-packages/`,
Python will also search the folder you cloned `./huggingface_hub/`.

## Install with conda

If you are more familiar with it, you can install `huggingface_hub` using the [conda-forge channel](https://anaconda.org/conda-forge/huggingface_hub):


```bash
conda install -c conda-forge huggingface_hub
```

Once done, [check installation](#check-installation) is working correctly.

## Check installation

Once installed, check that `huggingface_hub` works properly by running the following command:

```bash
python -c "from huggingface_hub import model_info; print(model_info('gpt2'))"
```

This command will fetch information from the Hub about the [gpt2](https://huggingface.co/gpt2) model.
Output should look like this:

```text
Model Name: gpt2
Tags: ['pytorch', 'tf', 'jax', 'tflite', 'rust', 'safetensors', 'gpt2', 'text-generation', 'en', 'doi:10.57967/hf/0039', 'transformers', 'exbert', 'license:mit', 'has_space']
Task: text-generation
```

## Windows limitations

With our goal of democratizing good ML everywhere, we built `huggingface_hub` to be a
cross-platform library and in particular to work correctly on both Unix-based and Windows
systems. However, there are a few cases where `huggingface_hub` has some limitations when
run on Windows. Here is an exhaustive list of known issues. Please let us know if you
encounter any undocumented problem by opening [an issue on Github](https://github.com/huggingface/huggingface_hub/issues/new/choose).

- `huggingface_hub`'s cache system relies on symlinks to efficiently cache files downloaded
from the Hub. On Windows, you must activate developer mode or run your script as admin to
enable symlinks. If they are not activated, the cache-system still works but in an non-optimized
manner. Please read [the cache limitations](./guides/manage-cache#limitations) section for more details.
- Filepaths on the Hub can have special characters (e.g. `"path/to?/my/file"`). Windows is
more restrictive on [special characters](https://learn.microsoft.com/en-us/windows/win32/intl/character-sets-used-in-file-names)
which makes it impossible to download those files on Windows. Hopefully this is a rare case.
Please reach out to the repo owner if you think this is a mistake or to us to figure out
a solution.


## Next steps

Once `huggingface_hub` is properly installed on your machine, you might want
[configure environment variables](package_reference/environment_variables) or [check one of our guides](guides/overview) to get started.
