# `huggingface_hub`

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://codecov.io/gh/huggingface/huggingface_hub"><img alt="Code coverage" src="https://codecov.io/gh/huggingface/huggingface_hub/branch/main/graph/badge.svg?token=RXP95LE2XL"></a>
<a href="https://github.com/huggingface/huggingface_hub/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/huggingface_hub.svg"></a>
<a href="https://huggingface.co/docs/huggingface_hub/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/huggingface_hub/index.svg?down_color=red&down_message=offline&up_message=online&label=doc"></a>

## Welcome to the huggingface_hub library


The `huggingface_hub` is a client library to interact with the Hugging Face Hub. The Hugging Face Hub is a platform with over 35K models, 4K datasets, and 2K demos in which people can easily collaborate in their ML workflows. The Hub works as a central place where anyone can share, explore, discover, and experiment with open-source Machine Learning.

With `huggingface_hub`, you can easily download and upload models, datasets, and Spaces. You can extract useful information from the Hub, and do much more. Some example use cases:
* Downloading and caching files from a Hub repository.
* Creating repositories and uploading an updated model every few epochs.
* Extract metadata from all models that match certain criteria (e.g. models for `text-classification`).
* List all files from a specific repository.

Read all about it in [the library documentation](https://huggingface.co/docs/huggingface_hub).

<br>

## Integrating to the Hub.

We're partnering with cool open source ML libraries to provide free model hosting and versioning. You can find the existing integrations [here](https://huggingface.co/docs/hub/libraries).

The advantages are:

- Free model or dataset hosting for libraries and their users.
- Built-in file versioning, even with very large files, thanks to a git-based approach.
- Hosted inference API for all models publicly available.
- In-browser widgets to play with the uploaded models.
- Anyone can upload a new model for your library, they just need to add the corresponding tag for the model to be discoverable.
- Fast downloads! We use Cloudfront (a CDN) to geo-replicate downloads so they're blazing fast from anywhere on the globe.
- Usage stats and more features to come.

If you would like to integrate your library, feel free to open an issue to begin the discussion. We wrote a [step-by-step guide](https://huggingface.co/docs/hub/adding-a-library) with ❤️ showing how to do this integration.

<br>

## Feedback (feature requests, bugs, etc.) is super welcome 💙💚💛💜♥️🧡
