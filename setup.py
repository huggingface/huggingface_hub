from setuptools import find_packages, setup
"""
Check list to release a new version.

1. Checkout the main branch and pull the latest changes:
   ```
   git checkout main
   git pull
   ```

2. Checkout a new branch with the version that you'd like to release: v<MINOR-VERSION>-release,
   for example `v0.5-release`. All patches will be done to that same branch.

3. Update the `__version__` variable in the `src/huggingface_hub/__init__.py` file to point 
   to the version you're releasing:
   ```
   __version__ = "<VERSION>"
   ```
   
4. Make sure that the conda build works correctly by building it locally:
   ```
   conda install -c defaults anaconda-client conda-build
   HUB_VERSION=<VERSION> conda-build .github/conda
   ```

5. Make sure that the pip wheel works correctly by building it locally and installing it:
   ```
   pip install setuptools wheel
   python setup.py sdist bdist_wheel
   pip install dist/huggingface_hub-<VERSION>-py3-none-any.whl
   ```
   
6. Commit, tag, and push the branch:
   ```
   git commit -am "Release: v<VERSION>"
   git tag v<VERSION> -m "Adds tag v<VERSION> for pypi and conda"
   git push -u --tags origin v<MINOR-VERSION>-release
   ```

7. Verify that the docs have been built correctly. You can check that on the following link:
   https://huggingface.co/docs/huggingface_hub/v<VERSION>
   
8. Checkout main once again to update the version in the `__init__.py` file:
   ```
   git checkout main
   ```

9. Update the version to contain the `.dev0` suffix:
   ```
   __version__ = "<VERSION+1>.dev0"  # For example, after releasing v0.5.0 or v0.5.1: "0.6.0.dev0".
   ```
   
10. Push the changes!
    ```
    git push origin main
    ```
"""


def get_version() -> str:
    rel_path = "src/huggingface_hub/__init__.py"
    with open(rel_path, "r") as fp:
        for line in fp.read().splitlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


install_requires = [
    "filelock",
    "requests",
    "tqdm",
    "pyyaml",
    "typing-extensions>=3.7.4.3",  # to be able to import TypeAlias
    "importlib_metadata;python_version<'3.8'",
    "packaging>=20.9",
]

extras = {}

extras["torch"] = [
    "torch",
]

extras["tensorflow"] = ["tensorflow", "pydot", "graphviz"]

extras["testing"] = [
    "pytest",
    "datasets",
    "soundfile",
]

extras["quality"] = [
    "black~=22.0",
    "isort>=5.5.4",
    "flake8>=3.8.3",
]

extras["all"] = extras["testing"] + extras["quality"]

extras["dev"] = extras["all"]


setup(
    name="huggingface_hub",
    version=get_version(),
    author="Hugging Face, Inc.",
    author_email="julien@huggingface.co",
    description="Client library to download and publish models on the huggingface.co hub",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="model-hub machine-learning models natural-language-processing deep-learning pytorch pretrained-models",
    license="Apache",
    url="https://github.com/huggingface/huggingface_hub",
    package_dir={"": "src"},
    packages=find_packages("src"),
    extras_require=extras,
    entry_points={
        "console_scripts": [
            "huggingface-cli=huggingface_hub.commands.huggingface_cli:main"
        ]
    },
    python_requires=">=3.7.0",
    install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
