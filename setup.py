import sys

from setuptools import find_packages, setup


def get_version() -> str:
    rel_path = "src/huggingface_hub/__init__.py"
    with open(rel_path, "r") as fp:
        for line in fp.read().splitlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


# hf-xet version used in both install_requires and extras["hf_xet"]
HF_XET_VERSION = "hf-xet>=1.2.0,<2.0.0"

install_requires = [
    "filelock",
    "fsspec>=2023.5.0",
    f"{HF_XET_VERSION}; platform_machine=='x86_64' or platform_machine=='amd64' or platform_machine=='AMD64' or platform_machine=='arm64' or platform_machine=='aarch64'",
    "httpx>=0.23.0, <1",
    "packaging>=20.9",
    "pyyaml>=5.1",
    "tqdm>=4.42.1",
    "typer",
    "typing-extensions>=4.1.0",  # to be able to import TypeAlias, dataclass_transform
]

extras = {}

extras["oauth"] = [
    "authlib>=1.3.2",  # minimum version to include https://github.com/lepture/authlib/pull/644
    "fastapi",
    "httpx",  # required for authlib but not included in its dependencies
    "itsdangerous",  # required for starlette SessionMiddleware
]

extras["torch"] = [
    "torch",
    "safetensors[torch]",
]
extras["fastai"] = [
    "toml",
    "fastai>=2.4",
    "fastcore>=1.3.27",
]

extras["hf_xet"] = [HF_XET_VERSION]

extras["mcp"] = ["mcp>=1.8.0"]

extras["testing"] = (
    extras["oauth"]
    + [
        "jedi",
        "Jinja2",
        "pytest>=8.4.2",  # we need https://github.com/pytest-dev/pytest/pull/12436
        "pytest-cov",
        "pytest-env",
        "pytest-xdist",
        "pytest-vcr",  # to mock Inference
        "pytest-asyncio",  # for AsyncInferenceClient
        "pytest-rerunfailures<16.0",  # to rerun flaky tests in CI
        "pytest-mock",
        "urllib3<2.0",  # VCR.py broken with urllib3 2.0 (see https://urllib3.readthedocs.io/en/stable/v2-migration-guide.html)
        "soundfile",
        "Pillow",
        "numpy",  # for embeddings
        "fastapi",  # To build the documentation
    ]
)

if sys.version_info >= (3, 10):
    # We need gradio to test webhooks server
    # But gradio 5.0+ only supports python 3.10+ so we don't want to test earlier versions
    extras["gradio"] = [
        "gradio>=5.0.0",
        "requests",  # see https://github.com/gradio-app/gradio/pull/11830
    ]

# Typing extra dependencies list is duplicated in `.pre-commit-config.yaml`
# Please make sure to update the list there when adding a new typing dependency.
extras["typing"] = [
    "typing-extensions>=4.8.0",
    "types-PyYAML",
    "types-simplejson",
    "types-toml",
    "types-tqdm",
    "types-urllib3",
]

extras["quality"] = [
    "ruff>=0.9.0",
    "mypy==1.15.0",
    "libcst>=1.4.0",
    "ty",
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
        "console_scripts": [
            "hf=huggingface_hub.cli.hf:main",
            "tiny-agents=huggingface_hub.inference._mcp.cli:app",
        ],
        "fsspec.specs": "hf=huggingface_hub.HfFileSystem",
    },
    python_requires=">=3.9.0",
    install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
    package_data={"huggingface_hub": ["py.typed"]},  # Needed for wheel installation
)
