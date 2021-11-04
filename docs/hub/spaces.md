---
title: Documentation for Spaces
---

<h1>How to get started with Spaces</h1>

## What are Spaces?

Spaces are a simple way to host ML demo apps directly on your profile or your organization’s  profile. This allows you to create your ML portfolio, showcase your projects at conferences or to stakeholders, and work collaboratively with other people in the ML ecosystem.

We support two awesome SDKs that let you build cool apps in Python in a matter of minutes: **[Streamlit](https://streamlit.io/)** and **[Gradio](https://gradio.app/)**.

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/3bSVKNKb_PY" title="Spaces intro" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

**To get started**, simply click on [New Space](https://huggingface.co/new-space) in the top navigation menu, create a new repo of type `Space`, and pick your SDK.

Under the hood, Spaces stores your code inside a git repository, just like the model and dataset repositories. Thanks to this, the same tools you're already used to (`git` and `git-lfs`) also work for Spaces.

## Should I use Streamlit or Gradio?

We recommend you try both because they're really awesome! 😎

Streamlit's documentation is at https://docs.streamlit.io/ and Gradio's doc is at https://gradio.app/getting_started.

In the default environment, we're currently running version `"1.0.0"` of Streamlit and the latest version of Gradio.

See [Configuration](#configuration) section for more infos on SDK versions.

Our 2 cents:

- **Gradio** is great if you want to build a super-easy-to-use interface to run a model from just the list of its inputs and its outputs. The Gradio team wrote a great [tutorial on our blog about building GUIs for Hugging Face models](https://huggingface.co/blog/gradio).
- **Streamlit** gives you more freedom to build a full-featured Web app from Python, in a _reactive_ way (meaning that code gets re-run when the state of the app changes). We wrote a short [blog post](https://huggingface.co/blog/streamlit-spaces) about using models and datasets with Spaces using Streamlit.

You can also take a look at some sample apps on the [Spaces directory](https://huggingface.co/spaces) to make up your mind.

[![screenshot of listing directory and landing page](/docs/assets/hub/spaces-landing.png)](https://huggingface.co/spaces)

If Streamlit and Gradio don't suit your needs, please get in touch with us. We're working on providing mechanisms to run **custom apps** with custom Python server code and a unified set of frontend JS code. Docker image serving is also on the works. If this sounds interesting, [reach out to us]((#how-can-i-contact-you)).

## What are the pre-installed dependencies in the default environment?

In addition to the Streamlit or Gradio SDK, the environment we run your app in includes the following Python libraries out-of-the-box:

- [`huggingface_hub`](https://github.com/huggingface/huggingface_hub), so you can download files (such as models) from the Hub, query the hf.co API, etc. 

**You can also use this to call our Accelerated Inference API from your Space**. If your app instantiates a model to run inference on, consider calling the Inference API instead, because you'll then leverage the acceleration optimizations we already built. This will also consuming less computing resources, which is always nice 🌎. See this [page](/docs/hub/how-to-inference) for more information on how to programmatically access the Inference API.

- [`requests`](https://docs.python-requests.org/en/master/) the famous HTTP request library, useful if you want to call a third-party API from your app.
- [`datasets`](https://github.com/huggingface/datasets) so that you can easily fetch or display data from inside your app.

## How can I install other dependencies?

If you need any other Python package, you can simply add a `requirements.txt` at the root of your repo.

A custom environment will be created on the fly by the Spaces runtime engine.

We also support Debian dependencies: add a `packages.txt` file at the root of your repo and list all your dependencies, one per line (each line will go through `apt-get install`)

## What are the RAM and CPU or GPU limitations?

Each environment is currently limited to 16GB RAM and 8 CPU cores.

For Pro or Organization (Lab or Startup plan) subscribers, Spaces can have one T4 GPU on a case-by-case basis, [contact us](#how-can-i-contact-you) if you need one.

## How does it work?

We deploy a containerized version of your code on our Infra, each time you commit. As a sidenote, we have many cool infra challenges to solve, if you'd like to help us, please consider [reaching out](#how-can-i-contact-you)!

## Secret management

If your app needs any secret keys or tokens to run, you do not want to hardcode them inside your code! Instead, head over to the settings page of your Space repo, and you'll be able to input key/secret pairs.

Secrets will be exposed to your app using the [Streamlit Secrets](https://blog.streamlit.io/secrets-in-sharing-apps/) feature if it's a Streamlit app, or as environment variables in other cases.


## I am having issues with Streamlit versions!

The Streamlit version is not configured in the `requirements.txt` file, but rather in the README metadata config through the `sdk_version` setting. Not all Streamlit versions are supported. Refer to the [reference section](#reference) for more information about which versions are supported.

## Can I use my own HTML instead of Streamlit or Gradio?

Although we strongly encourage you to use Streamlit and Gradio, you can also use your own HTML
code by defining `sdk: static` and having the HTML within an `index.html` file. Here are some examples:

* [Smarter NPC](https://huggingface.co/spaces/mishig/smarter_npc): Display a PlayCanvas project with an iframe.
* [Huggingfab](https://huggingface.co/spaces/pierreant-p/huggingfab): Display a Sketchfab model in Spaces.

Please [get in touch](#how-can-i-contact-you) if you have an idea for cool static Spaces.


## Building an organization card

Create an organization card to help users learn more about what your organization is working on and how users can use your libraries, models, datasets, and Spaces. Build an organization card by creating a static README Space with HTML. As an example, take a look at the [Amazon](https://huggingface.co/spaces/amazon/README/blob/main/README.md) and [spaCy](https://huggingface.co/spaces/spacy/README/blob/main/README.md) organization cards.

* https://huggingface.co/spaces/spacy/README/blob/main/README.md
* https://huggingface.co/spaces/amazon/README/blob/main/README.md


## Can I use Bokeh?

Streamlit has built-in support for Bokeh with the `st.bokeh_chart` component.

## How should I link my Spaces demo in my GitHub repo?

We have a badge that you can use, just replace the linked url with the correct one:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/your_user/your_space)

```
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/your_user/your_space)
```

## Can I use the Spaces logo to link to my app from my website?

That would be great! Here's the logo in SVG:

<img style="width: 280px;" src="/docs/assets/hub/icon-space.svg">

## Why did you build this?

In the past few years, our team, in collaboration with other research groups, has built a number of demo apps for some cool new models or methods (PPLM, RAG, zero-shot, ExBERT, etc.).

We host [widgets](https://huggingface-widgets.netlify.app/) for every model on the Hub, but in some cases (for instance if you want to compare two models) there is a need for a demo app that can't simply be implemented in a widget, so we needed something more flexible.

This project's goal is to build an extensible way for users and organizations to host demos/apps on huggingface.co in a more productized and scalable way.

## Configuration

All the settings of your Space are stored inside a YAML block on top of the `README.md` file at the root of the repository.

To modify those settings, you can edit this file, either by pushing to the repo via command-line, or directly on the hub

Sample `README.md` file :
```Markdown
---
title: Demo Space
emoji: 🤗
colorFrom: yellow
colorTo: orange
sdk: gradio
app_file: app.py
pinned: false
---
```

### Reference

**`title`** : _string_  
Display title for the Space

**`emoji`** : _string_  
Space emoji (emoji-only character allowed)

**`colorFrom`** : _string_  
Color for Thumbnail gradient (red, yellow, green, blue, indigo, purple, pink, gray)

**`colorTo`** : _string_  
Color for Thumbnail gradient (red, yellow, green, blue, indigo, purple, pink, gray)

**`sdk`** : _string_  
Can be either `gradio`, `streamlit` or `static`

**`sdk_version`** : _string_  
Only applicable for `streamlit` SDK. Currently available versions are :  
`0.79.0, 0.80.0, 0.81.1, 0.82.0, 0.83.0, 0.84.2, 0.85.0, 0.86.0, 0.87.0, 0.88.0, 0.89.0, 1.0.0`

**`app_file`** : _string_  
Path to your main application file (which contains either `gradio` or `streamlit` Python code).  
Path is relative to the root of the repository.

**`pinned`** : _boolean_  
Whether the Space stays on top of your list.

## How can I manage my app through Github

Keep your app in sync with your GitHub repository with GitHub Actions:

- We require Git LFS for files above 10MB so you may need to review your files if you don't want to use Git LFS. This includes your history. You can use handy tools such as [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/) to remove the large files from your history (keep a local copy of your repository for backup).
- Set your GitHub repository and your Spaces app initially in sync: to add your Spaces app as an additional remote to your existing git repository, you can use the command `git remote add space https://huggingface.co/spaces/FULL_SPACE_NAME`. You can then force-push to sync everything for the first time: `git push --force space main`
- Set up a GitHub Action to push your GitHub main branch automatically to Spaces: replace `HF_USERNAME` with your Hugging Face username, `FULL_SPACE_NAME` with your Spaces name, and [create a Github secret](https://docs.github.com/en/actions/reference/encrypted-secrets#creating-encrypted-secrets-for-an-environment) `HF_TOKEN` containing your Hugging Face API token.

```yaml
name: Sync to Hugging Face hub

on:
  push:
    branches: [main]

  # to run this workflow manually from the Actions tab
  workflow_dispatch:

jobs:
  sync-to-hub:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
        with:
          fetch-depth: 0
      - name: Push to hub
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: git push https://HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/FULL_SPACE_NAME main
```

- Create an action so file sizes are automatically checked on any new PR

```yaml
name: Check file size

on:               # or directly `on: [push]` to run the action on every push on any branch
  pull_request:
    branches: [main]

  # to run this workflow manually from the Actions tab
  workflow_dispatch:

jobs:
  sync-to-hub:
    runs-on: ubuntu-latest
    steps:
      - name: Check large files
        uses: ActionsDesk/lfs-warning@v2.0
        with:
          filesizelimit: 10485760 # = 10MB, so we can sync to HF spaces

```

## How can I contact you?

Feel free to ask questions on the [forum](https://discuss.huggingface.co/) if it's suitable for the community.

If you're interested in infra challenges, custom demos, GPUs, or something else, please reach out to us by sending an email to **website at huggingface.co**.

You can also tag us [on Twitter](https://twitter.com/huggingface)!

## Changelog

#### [2021-10-20] - Add support for Streamlit 1.0
- We now support all versions between 0.79.0 and 1.0.0

#### [2021-09-07] - Streamlit version pinning
- You can now choose which version of Streamlit will be installed within your Space

#### [2021-09-06] - Upgrade Streamlit to `0.84.2`
- Supporting Session State API
- [Streamlit changelog](https://github.com/streamlit/streamlit/releases/tag/0.84.0)

#### [2021-08-10] - Upgrade Streamlit to `0.83.0`
- [Streamlit changelog](https://github.com/streamlit/streamlit/releases/tag/0.83.0)

#### [2021-08-04] - Debian packages
- You can now add your `apt-get` dependencies into a `packages.txt` file

#### [2021-08-03] - Streamlit components
- Add support for [Streamlit components](https://streamlit.io/components)

#### [2021-08-03] - Flax/Jax GPU improvements
- For GPU-activated Spaces, make sure Flax / Jax runs smoothly on GPU

#### [2021-08-02] - Upgrade Streamlit to `0.82.0`
- [Streamlit changelog](https://github.com/streamlit/streamlit/releases/tag/0.82.0)

#### [2021-08-01] - Raw logs available
- Add link to raw logs (build and container) from the space repository (viewable by users with write access to a Space)
