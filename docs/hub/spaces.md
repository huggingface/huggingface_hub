---
title: Documentation for Spaces
---

<h1>How to get started with Spaces</h1>

## What are Spaces?

Spaces are a simple way to host ML demo apps directly on your profile or your organization’s  profile. This allows you to create your ML portfolio, showcase your projects at conferences or to stakeholders, and work collaboratively with other people in the ML ecosystem.

We support two awesome SDKs that let you build cool apps in Python in a matter of minutes: **[Streamlit](https://streamlit.io/)** and **[Gradio](https://gradio.app/)**.

**To get started**, simply click on [New Space](https://huggingface.co/new-space) in the top navigation menu, create a new repo of type `Space`, and pick your SDK.

Under the hood, Spaces stores your code inside a git repository, just like the model and dataset repositories. Thanks to this, the same tools you're already used to (`git` and `git-lfs`) also work for Spaces.

The default Spaces environment comes with several pre-installed dependencies:

* [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) allows you to download models from the Hub and programmatically access the Inference API from your Space. If you choose to instantiate the model in your app with our Inference API, you can benefit from the built-in acceleration optimizations. This option also consumes less computing resources, which is always nice for the environment! 🌎 

  Refer to this [page](https://huggingface.co/docs/hub/how-to-inference) for more information on how to programmatically access the Inference API.

* [`requests`](https://docs.python-requests.org/en/master/) is useful for calling third-party APIs from your app.

* [`datasets`](https://github.com/huggingface/datasets) allows you to fetch or display datasets from inside your app easily.

Each Spaces environment is limited to 16GB RAM and 8 CPU cores. Organization [subscribers](https://huggingface.co/pricing) (Lab, Startup, and Enterprise) can access Spaces with one T4 GPU on a case-by-case basis. Please email us at **website at huggingface.co** or let us know on [Twitter](https://twitter.com/huggingface) if you need one.

## Streamlit and Gradio

Spaces support [Streamlit](https://streamlit.io/) and [Gradio](https://gradio.app/) for quickly building apps in Python. The default environment runs version `"1.0.0"` of Streamlit and the latest version of Gradio. We recommend you try both because they're really awesome! Here are some of our thoughts on Streamlit and Gradio:

* **Gradio** provides an easy and intuitive interface for running a model from a list of inputs, and displaying the outputs. For more details, take a look at this [tutorial](https://huggingface.co/blog/gradio) from the Gradio team about building GUIs for Hugging Face models.

* **Streamlit** gives users more freedom to build a full-featured web app with Python in a *reactive* way. Your code is rerun each time the state of the app changes. Streamlit is also great for data visualization and supports several charting libraries such as Bokeh, Plotly, and Altair. Read our [blog post](https://huggingface.co/blog/streamlit-spaces) about building and hosting Streamlit apps in Spaces.

💡 If you need want to learn more about Streamlit and Gradio, refer to the [Streamlit documentation](https://docs.streamlit.io/) and [Gradio documentation](https://gradio.app/getting_started).

For more inspiration, take a look at some of the sample apps in the [Spaces directory](https://huggingface.co/spaces) to get a better idea of what Streamlit and Gradio can do.

[![screenshot of listing directory and landing page](/docs/assets/hub/spaces-landing.png)](https://huggingface.co/spaces)

## Using Spaces

Create a Space by clicking on [New Space](https://huggingface.co/new-space) under your profile picture in the top navigation bar. Next, create a repository of type `Space`, and then you can select whether you want to use Streamlit or Gradio.

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/3bSVKNKb_PY" title="Spaces intro" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Install other dependencies

If you need other Python packages to run your app, add it to a **requirements.txt** file at the root of your repository. Spaces runtime engine will create a custom environment on-the-fly. 

Debian dependencies are also supported. Add a **packages.txt** file at the root of your repository, and list all your dependencies in it. Each dependency should be on a separate line, and each line will be read and installed by `apt-get install`.

### Manage secrets

If your app requires secret keys or tokens, don't hard-code them inside your app! Instead, go to the **Settings** page of your Space repository and enter your secrets there. The secrets will be exposed to your app with [Streamlit Secrets Management](https://blog.streamlit.io/secrets-in-sharing-apps/) if you use Streamlit, and as environment variables in other cases. 

![screenshot of secrets settings](/docs/assets/hub/secrets.png)

### Custom HTML

Spaces also accommodate custom HTML for your app instead of using Streamlit or Gradio. Set `sdk: static` inside the `YAML` block at the top of your Spaces **README.md** file. Then you can place your HTML code within an **index.html** file.

Here are some examples of Spaces using custom HTML:

* [Smarter NPC](https://huggingface.co/spaces/mishig/smarter_npc): Display a PlayCanvas project with an iframe in Spaces.
* [Huggingfab](https://huggingface.co/spaces/pierreant-p/huggingfab): Display a Sketchfab model in Spaces.

💌 Feel free to [contact us](#contact) if you are interested in building custom apps without Streamlit or Gradio. Our team is working on creating mechanisms for running custom apps with Python server code with a unified set of frontend JS code and serving Docker images.

### Configure Spaces settings

Configure your Space's appearance and other settings inside the `YAML` block at the top of the **README.md** file at the root of the repository. For example, if you want to create a Space with Gradio named `Demo Space` with a yellow to orange gradient thumbnail:

```yaml
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

For additional settings, refer to the [Reference](#reference) section.

### Organization card

Create an organization card to help users learn more about what your organization is working on and how users can use your libraries, models, datasets, and Spaces. Build an organization card by setting `sdk: static` to create a static **README** Space with HTML. As an example, take a look at:

* [Amazon's](https://huggingface.co/spaces/amazon/README/blob/main/README.md) organization card
* [spaCy's](https://huggingface.co/spaces/spacy/README/blob/main/README.md) organization card.

### Manage app with Github Actions

Keep your app in sync with your Github repository with Github Actions. For files larger than 10MB, Spaces requires Git-LFS. If you don't want to use Git-LFS, you may need to review your files and check your history. Use a tool like [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/) to remove any large files from your history. BFG Repo-Cleaner will keep a local copy of your repository as a backup.

First, you should setup your GitHub repository and Spaces app together. Add your Spaces app as an additional remote to your existing Git repository.

```bash
git remote add space https://huggingface.co/spaces/FULL_SPACE_NAME
```

Then force push to sync everything for the first time:

```bash
git push --force space main
```

Next, setup a GitHub Action to push your main branch to Spaces. In the example below:

* Replace `HF_USERNAME` with your username and `FULL_SPACE_NAME` with your Space name. 
* Create a [Github secret](https://docs.github.com/en/actions/security-guides/encrypted-secrets#creating-encrypted-secrets-for-an-environment) with your `HF_TOKEN`. You can find your Hugging Face API token under **API Tokens** on your Hugging Face profile.

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

Finally, create an Action that automatically checks the file size of any new pull request:


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
          filesizelimit: 10485760 # this is 10MB so we can sync to HF Spaces
```

## Troubleshoot

Issues may occur when you use an unsupported Streamlit version. The Streamlit version is not configured in the **requirements.txt** file but rather in the `YAML` settings through the `sdk_version` setting. Not all Streamlit versions are supported. Check that you are using a supported version of Streamlit. Refer to the [reference section](#reference) for more information about supported versions.

## Contact

Feel free to ask questions on the [forum](https://discuss.huggingface.co/) if it's suitable for the community.

If you're interested in infra challenges, custom demos, GPUs, or something else, please reach out to us by sending an email to **website at huggingface.co**.

You can also tag us [on Twitter](https://twitter.com/huggingface)! 🤗

---

## Reference

**`title`** : _string_
Display title for the Space.

**`emoji`** : _string_
Space emoji (emoji-only character allowed).

**`colorFrom`** : _string_
Color for Thumbnail gradient (red, yellow, green, blue, indigo, purple, pink, gray).

**`colorTo`** : _string_
Color for Thumbnail gradient (red, yellow, green, blue, indigo, purple, pink, gray).

**`sdk`** : _string_
Can be either `gradio`, `streamlit` or `static`.

**`sdk_version`** : _string_
Only applicable for `streamlit` SDK. Currently available versions are :  
`0.79.0, 0.80.0, 0.81.1, 0.82.0, 0.83.0, 0.84.2, 0.85.0, 0.86.0, 0.87.0, 0.88.0, 0.89.0, 1.0.0`.

**`app_file`** : _string_
Path to your main application file (which contains either `gradio` or `streamlit` Python code, or `static` html code).
Path is relative to the root of the repository.

**`pinned`** : _boolean_
Whether the Space stays on top of your list.

---

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
