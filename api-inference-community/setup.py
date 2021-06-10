from setuptools import setup


setup(
    name="api_inference_community",
    version="0.0.7",
    description="A package with helper tools to build an API Inference docker app for Hugging Face API inference using huggingface_hub",
    url="http://github.com/huggingface/api-inference-community",
    author="Nicolas Patry",
    author_email="nicolas@huggingface.co",
    license="MIT",
    packages=["api_inference_community"],
    python_requires=">=3.6.0",
    zip_safe=False,
    install_requires=list(line for line in open("requirements.txt", "r")),
)
