---
title: Documentation for Spaces
---

<h1>How to get started with Spaces</h1>

> ⚠️ This feature is currently in private beta, reach out to [julien-c](https://huggingface.co/julien-c) if you'd like to try it out ⚠️

## What are Spaces?

Spaces are a simple way to host a ML demo app, directly on your user profile or your organization’s hf.co profile.

We support two awesome SDKs that let you build cool apps in Python: **[Streamlit](https://streamlit.io/)** and **[Gradio](https://gradio.app/)**.

**To get started**, simply click on [New Space](https://huggingface.co/new-space) in the top navigation menu, create a new repo of type `Space`, and pick your SDK:

![/docs/assets/hub/new-space.gif](/docs/assets/hub/new-space.gif)

Under the hood, we are storing your code inside a hf.co-hosted git repository, similar to what we're doing for models and datasets. So the same tools you're already used to (`git` and `git-lfs`) will also work for Spaces.

We then deploy a containerized version of your code on our Infra, each time you commit. More details below!

<!-- TODO(have someone record a Youtube demo of Spaces showcasing some cool apps already running, etc) -->

## Should I use Streamlit or Gradio?

We recommend you try both as they're both really awesome! 😎

Streamlit's documentation is at https://docs.streamlit.io/, and Gradio's doc is https://gradio.app/docs.

In the default environment, we're currently running version `"0.79.0"` of Streamlit and version `"2.0.9"` of Gradio.

Our 2 cents:
- **Gradio** is great if you want to build a super-easy-to-use interface to run a model from just the list of its inputs and its outputs. The Gradio team wrote a great [tutorial on our blog about building GUIs for Hugging Face models](https://huggingface.co/blog/gradio).
- **Streamlit** gives you more freedom to build a full-featured Web app from Python, in a _reactive_ way (meaning that code gets re-run when the state of the app changes).

You can also take a look at some sample apps on the [Spaces directory](https://huggingface.co/spaces) (⚠️ Note: not live yet!) to make up your mind.

<!-- Add screencap of listing directory -->

Finally, we've been thinking of providing a way to run **custom apps**, for instance Python server code for the backend + a unified set of widgets/frontend JS code, or even custom Docker image serving. Do get in touch if you would like to build something more custom.


## What are the pre-installed dependencies in the default environment?

In addition to the Streamlit or Gradio SDK, the environment we run your app in includes the following Python libraries out-of-the-box:
- [`huggingface_hub`](https://github.com/huggingface/huggingface_hub), so you can list models, query the hf.co API, etc. **You can also use this to call our Accelerated Inference API from your Space**. If your app instantiates a model to run inference on, consider calling the Inference API instead, because you'll then leverage the acceleration optimizations we already built, and it's also consuming less computing resources, which is always nice 🌎.
<!-- TODO(merge and ship the Inference API wrapper) -->
- [`requests`](https://docs.python-requests.org/en/master/) the famous HTTP request library, useful if you want to call a third-party API from your app.
- [`datasets`](https://github.com/huggingface/datasets) so that you can easily fetch or display data from inside your app.

## How can I install other dependencies?

If you need any other Python package, you can simply add a `requirements.txt` at the root of your repo.

A custom environment will be created on the fly by the Spaces runtime engine.

We do not support installing `apt-get` dependencies yet, but it's on our roadmap.

## What are the RAM and CPU or GPU limitations?

Each environment is currently limited to 16GB RAM and 8 CPU cores.

Some Spaces can have one T4 GPU on a case-by-case basis, contact us if you need one.

## How does it work?

We deploy a containerized version of your code on our Infra, each time you commit. As a sidenote, we have many cool infra challenges to solve, if you'd like to help us, please consider reaching out 🙂.

## Secret management

If your app needs any secret keys or tokens to run, you do not want to hardcode them inside your code. Instead, head over to the settings page for your Space repo and you'll be able to input key/secret pairs.

Those secrets will be exposed to your app using the [Streamlit Secrets](https://blog.streamlit.io/secrets-in-sharing-apps/) feature if it's a Streamlit app, or as env variables in other cases.

## Streamlit advanced features

We support those Streamlit features transparently:
- `st.experimental_get_query_params()` and `st.experimental_set_query_params(**parameter)` to manage app state in the url
- if something doesn't work, please reach out.

## Can I use the Spaces logo to link to my app from my website?

Yes that would be great, here's the logo in SVG:

<img style="width: 280px;" src="/docs/assets/hub/icon-space.svg">


## Why did you build this?

In the past few years, our team, in collaboration with other research groups, has built a number of demo apps for some cool new models or methods (PPLM, RAG, zero-shot, ExBERT, etc.).

We host [widgets](https://huggingface-widgets.netlify.app/) for every model on the Hub, but in some cases (for instance if you want to compare two models) there is a need for a demo app that can't simply be implemented in a widget, so we needed something more flexible.

This project's goal is to experiment with an extensible way for users and organizations to host demos/apps on huggingface.co, in a more productized/scalable way than we’ve done in the past.
