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
    "fsspec>=2023.5.0",
    "hf-xet>=1.1.0,<2.0.0; platform_machine=='x86_64' or platform_machine=='amd64' or platform_machine=='arm64' or platform_machine=='aarch64'",
    "packaging>=20.9",
    "pyyaml>=5.1",
    "requests",
    "tqdm>=4.42.1",
    "typing-extensions>=3.7.4.3",  # to be able to import TypeAlias
]

extras = {}

extras["cli"] = [
    "InquirerPy==0.3.4",  # Note: installs `prompt-toolkit` in the background
]

extras["inference"] = [
    "aiohttp",  # for AsyncInferenceClient
]

extras["torch"] = [
    "torch",
    "safetensors[torch]",
]
extras["hf_transfer"] = [
    "hf_transfer>=0.1.4",  # Pin for progress bars
]
extras["fastai"] = [
    "toml",
    "fastai>=2.4",
    "fastcore>=1.3.27",
]

extras["tensorflow"] = [
    "tensorflow",
    "pydot",
    "graphviz",
]

extras["tensorflow-testing"] = [
    "tensorflow",
    "keras<3.0",
]

extras["hf_xet"] = ["hf_xet>=1.1.0,<2.0.0"]

extras["testing"] = (
    extras["cli"]
    + extras["inference"]
    + [
        "jedi",
        "Jinja2",
        "pytest>=8.1.1,<8.2.2",  # at least until 8.2.3 is released with https://github.com/pytest-dev/pytest/pull/12436
        "pytest-cov",
        "pytest-env",
        "pytest-xdist",
        "pytest-vcr",  # to mock Inference
        "pytest-asyncio",  # for AsyncInferenceClient
        "pytest-rerunfailures",  # to rerun flaky tests in CI
        "pytest-mock",
        "urllib3<2.0",  # VCR.py broken with urllib3 2.0 (see https://urllib3.readthedocs.io/en/stable/v2-migration-guide.html)
        "soundfile",
        "Pillow",
        "gradio>=4.0.0",  # to test webhooks # pin to avoid issue on Python3.12
        "numpy",  # for embeddings
        "fastapi",  # To build the documentation
    ]
)

# Typing extra dependencies list is duplicated in `.pre-commit-config.yaml`
# Please make sure to update the list there when adding a new typing dependency.
extras["typing"] = [
    "typing-extensions>=4.8.0",
    "types-PyYAML",
    "types-requests",
    "types-simplejson",
    "types-toml",
    "types-tqdm",
    "types-urllib3",
]

extras["quality"] = [
    "ruff>=0.9.0",
    "mypy==1.5.1",
    "libcst==1.4.0",
]

extras["all"] = extras["testing"] + extras["quality"] + extras["typing"]

extras["dev"] = extras["all"]

setup(
    name="huggingface_hub",
    version=get_version(),
    author="Hugging Face, Inc.",
    author_email="julien@huggingface.co",
    description="Client library to download and publish models, datasets and other repos on the huggingface.co hub",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="model-hub machine-learning models natural-language-processing deep-learning pytorch pretrained-models",
    license="Apache",
    url="https://github.com/huggingface/huggingface_hub",
    package_dir={"": "src"},
    packages=find_packages("src"),
    extras_require=extras,
    entry_points={
        "console_scripts": ["huggingface-cli=huggingface_hub.commands.huggingface_cli:main"],
        "fsspec.specs": "hf=huggingface_hub.HfFileSystem",
    },
    python_requires=">=3.8.0",
    install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
    package_data={"huggingface_hub": ["py.typed"]},  # Needed for wheel installation
)
