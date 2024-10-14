<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Installation

Before you start, you will need to setup your environment by installing the appropriate packages.

`huggingface_hub` is tested on **Python 3.8+**.

## Install with pip

It is highly recommended to install `huggingface_hub` in a [virtual environment](https://docs.python.org/3/library/venv.html).
If you are unfamiliar with Python virtual environments, take a look at this [guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).
A virtual environment makes it easier to manage different projects, and avoid compatibility issues between dependencies.

Start by creating a virtual environment in your project directory:

```bash
python -m venv .env
```

Activate the virtual environment. On Linux and macOS:

```bash
source .env/bin/activate
```

Activate virtual environment on Windows:

```bash
.env/Scripts/activate
```

Now you're ready to install `huggingface_hub` [from the PyPi registry](https://pypi.org/project/huggingface-hub/):

```bash
pip install --upgrade huggingface_hub
```

Once done, [check installation](#check-installation) is working correctly.

### Install optional dependencies

Some dependencies of `huggingface_hub` are [optional](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies) because they are not required to run the core features of `huggingface_hub`. However, some features of the `huggingface_hub` may not be available if the optional dependencies aren't installed.

You can install optional dependencies via `pip`:
```bash
# Install dependencies for tensorflow-specific features
# /!\ Warning: this is not equivalent to `pip install tensorflow`
pip install 'huggingface_hub[tensorflow]'

# Install dependencies for both torch-specific and CLI-specific features.
pip install 'huggingface_hub[cli,torch]'
```

Here is the list of optional dependencies in `huggingface_hub`:
- `cli`: provide a more convenient CLI interface for `huggingface_hub`.
- `fastai`, `torch`, `tensorflow`: dependencies to run framework-specific features.
- `dev`: dependencies to contribute to the lib. Includes `testing` (to run tests), `typing` (to run type checker) and `quality` (to run linters).



### Install from source

In some cases, it is interesting to install `huggingface_hub` directly from source.
This allows you to use the bleeding edge `main` version rather than the latest stable version.
The `main` version is useful for staying up-to-date with the latest developments, for instance
if a bug has been fixed since the last official release but a new release hasn't been rolled out yet.

However, this means the `main` version may not always be stable. We strive to keep the
`main` version operational, and most issues are usually resolved
within a few hours or a day. If you run into a problem, please open an Issue so we can
fix it even sooner!

```bash
pip install git+https://github.com/huggingface/huggingface_hub
```

When installing from source, you can also specify a specific branch. This is useful if you
want to test a new feature or a new bug-fix that has not been merged yet:

```bash
pip install git+https://github.com/huggingface/huggingface_hub@my-feature-branch
```

Once done, [check installation](#check-installation) is working correctly.

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
For example, if your Python packages are typically installed in `./.venv/lib/python3.12/site-packages/`,
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
