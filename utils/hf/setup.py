import os

from setuptools import setup


def get_version() -> str:
    rel_path = os.path.join(os.path.dirname(__file__), "../../src/huggingface_hub/__init__.py")
    with open(rel_path, "r") as fp:
        for line in fp.read().splitlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


install_requires = [
    f"huggingface_hub=={get_version()}",
]

setup(
    name="hf",
    version=get_version(),
    author="Hugging Face, Inc.",
    author_email="julien@huggingface.co",
    description="CLI extracted from the huggingface_hub library to interact with the Hugging Face Hub",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache",
    url="https://github.com/huggingface/huggingface_hub",
    packages=["hf"],  # dummy package to raise ImportError on import
    entry_points={"console_scripts": ["hf=huggingface_hub.cli.hf:main"]},
    python_requires=">=3.9.0",
    install_requires=install_requires,
    classifiers=[],
)
