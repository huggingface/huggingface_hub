from setuptools import find_packages, setup


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
    "pyyaml>=5.1",
    "typing-extensions>=3.7.4.3",  # to be able to import TypeAlias
    "importlib_metadata;python_version<'3.8'",
    "packaging>=20.9",
]

extras = {}

extras["cli"] = [
    "InquirerPy==0.3.4",
    # Note: installs `prompt-toolkit` in the background
]

extras["torch"] = [
    "torch",
]

extras["fastai"] = [
    "toml",
    "fastai>=2.4",
    "fastcore>=1.3.27",
]

extras["tensorflow"] = ["tensorflow", "pydot", "graphviz"]

extras["testing"] = extras["cli"] + [
    "isort>=5.5.4",
    "jedi",
    "Jinja2",
    "pytest",
    "pytest-cov",
    "pytest-env",
    "soundfile",
]

# Typing extra dependencies list is duplicated in `.pre-commit-config.yaml`
# Please make sure to update the list there when adding a new typing dependency.
extras["typing"] = [
    "types-PyYAML",
    "types-requests",
    "types-simplejson",
    "types-toml",
    "types-tqdm",
    "types-urllib3",
]

extras["quality"] = [
    "black==22.3",
    "flake8>=3.8.3",
    "flake8-bugbear",
    "isort>=5.5.4",
    "mypy==0.982",
]

extras["all"] = extras["testing"] + extras["quality"] + extras["typing"]

extras["dev"] = extras["all"]


setup(
    name="huggingface_hub",
    version=get_version(),
    author="Hugging Face, Inc.",
    author_email="julien@huggingface.co",
    description=(
        "Client library to download and publish models, datasets and other repos on the"
        " huggingface.co hub"
    ),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords=(
        "model-hub machine-learning models natural-language-processing deep-learning"
        " pytorch pretrained-models"
    ),
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
    include_package_data=True,
)
