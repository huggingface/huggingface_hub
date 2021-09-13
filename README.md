# Hugging Face Hub

## Welcome to the Hub repo! Here you can find all the open source things related to the Hugging Face Hub.

<p align="center">
	<img alt="Build" src="https://github.com/huggingface/huggingface_hub/workflows/Python%20tests/badge.svg">
	<a href="https://github.com/huggingface/huggingface_hub/blob/master/LICENSE">
		<img alt="GitHub" src="https://img.shields.io/github/license/huggingface/huggingface_hub.svg?color=blue">
	</a>
	<a href="https://github.com/huggingface/huggingface_hub/releases">
		<img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/huggingface_hub.svg">
	</a>
</p>

What can you find in this repo?

* [`huggingface_hub`](https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub), a client library to download and publish on the Hugging Face Hub as well as extracting useful information from there.
* [`api-inference-community`](https://github.com/huggingface/huggingface_hub/tree/main/api-inference-community), the Inference API for open source machine learning libraries.
* [`widgets`](https://github.com/huggingface/huggingface_hub/tree/main/widgets), the open sourced widgets that allow people to try out the models in the browser.
  * [`interfaces`](https://github.com/huggingface/huggingface_hub/tree/main/widgets/src/lib/interfaces), Typescript definition files for the Hugging Face Hub.
* [`docs`](https://github.com/huggingface/huggingface_hub/tree/main/docs), containing the official [Hugging Face Hub documentation](https://hf.co/docs).


## Quickstart

In this quick tour, we show you how to install the library and some of its core features.
### Installation

To install, run:

```bash
python -m pip install -U huggingface-hub
```
### Extracting information from the Hub

`huggingface_hub` provides a `HfApi` class that allows you to programatically interact with the Hub in various ways. For example, you can list all the public models on the Hub and apply filters via the `HfApi.list_models` function:

```python
from huggingface_hub import HfApi

api = HfApi()

# List all models
api.list_models()

# List only the text classification models
api.list_models(filter="text-classification")

# List only the russian models compatible with pytorch
api.list_models(filter=("ru", "pytorch"))

# List only the models trained on the "common_voice" dataset
api.list_models(filter="dataset:common_voice")

# List only the models from the spaCy library
api.list_models(filter="spacy")
```

Similarly, you can also inspect all the public datasets on the Hub via the `HfApi.list_datasets` function:

```python
from huggingface_hub import HfApi

api = HfApi()

# List only the text classification datasets
api.list_datasets(filter="task_categories:text-classification")

# List only the datasets in russian for language modeling
api.list_datasets(filter=("languages:ru", "task_ids:language-modeling"))
```

If you want to inspect the metadata of a single model or dataset, you can use the following functions:

```python
from huggingface_hub import HfApi

api = HfApi()

# Get metadata of a single model
api.model_info("distilbert-base-uncased")

# Get metadata of a single dataset
api.dataset_info("glue")
```

### Creating and deleting a repo

The `HfApi` class also allows you to create and delete repos, as well as change their visibility from public to private. Here is how it's done for model repos:

```python
from huggingface_hub import HfApi

api = HfApi()

# Create a public model repo
api.create_repo(token=YOUR_HF_API_TOKEN, name=REPO_NAME)

# Make it private
api.update_repo_visibility(token=YOUR_HF_API_TOKEN name=REPO_NAME, private=True)

# Delete the model repo - irreversible so be careful!
api.delete_repo(token=YOUR_HF_API_TOKEN name=REPO_NAME)
```

You can find your API token by clicking on your Hub profile and selecting _Settings > API Tokens_. To create and delete dataset repos, you simply need to provide an additional `type="dataset"` argument:

```python
from huggingface_hub import HfApi

api = HfApi()

# Create a public dataset repo
api.create_repo(token=YOUR_HF_API_TOKEN, name=REPO_NAME, type="dataset")
```

### Mixins to push your models to the Hub

`huggingface_hub` provides a `ModelHubMixin` that can be subclassed to create custom logic for saving and loading your models in any framework. See the source of `PyTorchModelHubMixin` which allows you to push PyTorch models to the Hub as follows:

```python
import torch
from huggingface_hub import PyTorchModelHubMixin

class MyModel(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Transformer()

model = MyModel()
# Pushes pytorch_model.bin to the Hub
model.push_to_hub(repo_path_or_name=REPO_NAME)
```

Once the model's files are pushed to the Hub, you can then reload from anywhere with

```python
model = MyModel()
model.from_pretrained(model_id=REPO_NAME)
```

## The `huggingface_hub` client library

This library allows anyone to work with the Hub repositories: you can clone them, create them and upload your models to them. On top of this, the library also offers methods to access information from the Hub. For example, listing all models that meet specific criteria or get all the files from a specific repo. You can find the library implementation [here](https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub).

<br>

## Integrating to the Hub.

We're partnering with cool open source ML libraries to provide free model hosting and versioning. You can find the existing integrations [here](https://huggingface.co/docs/hub/libraries).

The advantages are:

- Free model hosting for libraries and their users.
- Built-in file versioning, even with very large files, thanks to a git-based approach.
- Hosted inference API for all models publicly available.
- In-browser widgets to play with the uploaded models.
- Anyone can upload a new model for your library, they just need to add the corresponding tag for the model to be discoverable.
- Fast downloads! We use Cloudfront (a CDN) to geo-replicate downloads so they're blazing fast from anywhere on the globe.
- Usage stats and more features to come.

If you would like to integrate your library, feel free to open an issue to begin the discussion. We wrote a [step-by-step guide](https://huggingface.co/docs/hub/adding-a-library) with ❤️ showing how to do this integration.

<br>

## Inference API integration into the Hugging Face Hub

In order to get a functional Inference API on the Hub for your models (and thus, cool working widgets!) check out this [doc](https://github.com/huggingface/huggingface_hub/tree/main/api-inference-community). There is a docker image for each library. Within the image, you can find the implementation for supported pipelines for the given library.

<br>


## Widgets

All our widgets are open-sourced. Feel free to propose and implement widgets. You can try all of them out [here](https://huggingface-widgets.netlify.app/).


<br>

## Code Snippets

We'll implement a few tweaks to improve the UX for your models on the website – let's use [Asteroid](https://github.com/asteroid-team/asteroid) as an example.

Model authors add an `asteroid` tag to their model card and they get the advantages of model versioning built-in

![asteroid-model](docs/assets/asteroid_repo.png)

We add a custom "Use in Asteroid" button. When clicked, you get a library-specific code sample that you'll be able to specify. 🔥

![asteroid-code-sample](docs/assets/asteroid_snippet.png)


<br>

## Contributing

Feedback (feature requests, bugs, etc.) is super welcome 💙💚💛💜♥️🧡.

### Developer installation

To contribute a feature to `huggingface_hub`, first fork the repository by clicking the "Fork" button on the repository’s page. This creates a clone of the repository under your GitHub user. Next, clone the fork and install the project's requirements:

```bash
git clone git@github.com:{YOUR_GITHUB_USERNAME}/huggingface_hub.git
cd huggingface_hub
python -m pip install -e ".[dev]"
```
### Testing

Use the following environment variable to run the tests locally:

```bash
HUGGINGFACE_CO_STAGING=yes pytest -sv ./tests/
```

