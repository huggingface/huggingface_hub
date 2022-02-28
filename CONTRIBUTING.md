<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# How to contribute to huggingface_hub, the GitHub repository?

Everyone is welcome to contribute, and we value everybody's contribution. Code
is thus not the only way to help the community. Answering questions, helping
others, reaching out and improving the documentations are immensely valuable to
the community.

It also helps us if you spread the word: reference the library from blog posts
on the awesome projects it made possible, shout out on Twitter every time it has
helped you, or simply star the repo to say "thank you".

Whichever way you choose to contribute, please be mindful to respect our
[code of conduct](https://github.com/huggingface/huggingface_hub/blob/master/CODE_OF_CONDUCT.md).

## You can contribute in so many ways!

The repository is split into different parts, where we welcome contributions.

What can you find in this repo?

* [`huggingface_hub`](https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub), a client library to download and publish on the Hugging Face Hub as well as extracting useful information from there.
* [`api-inference-community`](https://github.com/huggingface/huggingface_hub/tree/main/api-inference-community), the Inference API for open source machine learning libraries.
* [`js`](https://github.com/huggingface/huggingface_hub/tree/main/js), the open sourced widgets that allow people to try out the models in the browser.
  * [`interfaces`](https://github.com/huggingface/huggingface_hub/tree/main/js/src/lib/interfaces), Typescript definition files for the Hugging Face Hub.
* [`docs`](https://github.com/huggingface/huggingface_hub/tree/main/docs), containing the official [Hugging Face Hub documentation](https://hf.co/docs).


### The client library, `huggingface_hub`

This repository hosts the client library `huggingface_hub`, which is a frontend to the Hugging Face Hub.
this part lives in `src/huggingface_hub` and `tests`.

There are many ways you can contribute to this client library:
* Fixing outstanding issues with the existing code;
* Contributing to the examples or to the documentation;
* Submitting issues related to bugs or desired new features.

When opening a PR on this part of the repository, please add as reviewer:
- @LysandreJik
- @osanseviero

Additionally, here are the owners of specific parts of the library:
- Core of the library: @muellerzr
- Documentation: @stevhliu
- Mixins: @nateraw

### The community inference API

The `api-inference-community` folder contains a tool to enable third-party library support integrated with the Hugging
Face Hub.

We welcome contributions to [add new containers](https://huggingface.co/docs/hub/adding-a-library#set-up-the-inference-api) for new libraries, to update the existing ones, and to provide help
fixing bugs and adding features. This folder contains an additional README
file explaining how you may test your code.

When opening a PR on this part of the repository, please add @Narsil as a reviewer.

## JavaScript content

The `js` folder contains the JavaScript code of the Hub. It includes:
* The widgets ([code](https://github.com/huggingface/huggingface_hub/tree/main/js/src/lib/components/InferenceWidget)
* Code snippets to make inference calls ([code](https://github.com/huggingface/huggingface_hub/tree/main/js/src/lib/inferenceSnippets)
* Code snippets to load models ([code](https://github.com/huggingface/huggingface_hub/blob/main/js/src/lib/interfaces/Libraries.ts))

Here too, we welcome any logic and documentation contributions. This folder contains an additional README
file explaining how you may test your code.

When opening a PR on this part of the repository, please add @mishig25 and @osanseviero as a reviewer.

## Documentation

The content in the `docs` folder is the official [Hugging Face Hub documentation]. It is not limited to the 
Python package `huggingface_hub`, as it includes guides on using the frontend, ho to build Spaces, how to search
efficiently, and others.

Here too, we welcome contribution, may it be for syntactic changes or typos.

*All contributions are equally valuable to the community.*

## Submitting a new issue or feature request

Do your best to follow these guidelines when submitting an issue or a feature
request. It will make it easier for us to come back to you quickly and with good
feedback.

### Did you find a bug?

The 🤗 Hugging Face Hub library is robust and reliable thanks to the users who notify us of
the problems they encounter. So thank you for reporting an issue.

First, we would really appreciate it if you could **make sure the bug was not
already reported** (use the search bar on Github under Issues).

Did not find it? :( So we can act quickly on it, please follow these steps:

* Include your **OS type and version**, the versions of **Python**, **PyTorch** and
  **Tensorflow** when applicable;
* A short, self-contained, code snippet that allows us to reproduce the bug in
  less than 30s;
* Provide the *full* traceback if an exception is raised by copying the text from your terminal 
  in the issue description.

### Do you want a new feature?

A good feature request addresses the following points:

1. Motivation first:
* Is it related to a problem/frustration with the library? If so, please explain
  why and provide a code snippet that demonstrates the problem best.
* Is it related to something you would need for a project? We'd love to hear
  about it!
* Is it something you worked on and think could benefit the community?
  Awesome! Tell us what problem it solved for you.
2. Write a *full paragraph* describing the feature;
3. Provide a **code snippet** that demonstrates its future use;
4. In case this is related to a paper, please attach a link;
5. Attach any additional information (drawings, screenshots, etc.) you think may help.

If your issue is well written we're already 80% of the way there by the time you
post it.

## Submitting a pull request (PR)

Before writing code, we strongly advise you to search through the existing PRs or
issues to make sure that nobody is already working on the same thing. If you are
unsure, it is always a good idea to open an issue to get some feedback.

You will need basic `git` proficiency to be able to contribute to
`huggingface_hub`. `git` is not the easiest tool to use but it has the greatest
manual. Type `git --help` in a shell and enjoy. If you prefer books, [Pro
Git](https://git-scm.com/book/en/v2) is a very good reference.

Follow these steps to start contributing:

1. Fork the [repository](https://github.com/huggingface/huggingface_hub) by
   clicking on the 'Fork' button on the repository's page. This creates a copy of the code
   under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote. The following command
   assumes you have your public SSH key uploaded to GitHub. See the following guide for more
   [information](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

   ```bash
   $ git clone git@github.com:<your Github handle>/huggingface_hub.git
   $ cd huggingface_hub
   $ git remote add upstream https://github.com/huggingface/huggingface_hub.git
   ```

3. Create a new branch to hold your development changes, and do this for every new PR you work on.

   Start by synchronizing your `main` branch with the `upstream/main` branch (ore details in the [GitHub Docs](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork)):

   ```bash
   $ git checkout main
   $ git fetch upstream
   $ git merge upstream/main
   ```

   Once your `main` branch is synchronized, create a new branch from it:

   ```bash
   $ git checkout -b a-descriptive-name-for-my-changes
   ```

   **Do not** work on the `master` branch.

4. Set up a development environment by running the following command in a virtual environment a conda or a 
   virtual environment you've created for working on this library:

   ```bash
   $ pip install -e ".[dev]"
   ```

   (If huggingface_hub was already installed in the virtual environment, remove
   it with `pip uninstall huggingface_hub` before reinstalling it in editable
   mode with the `-e` flag.)

5. Develop the features on your branch.

   As you work on the features, you should make sure that the test suite
   passes. You should run the tests impacted by your changes like this (see 
   below an explanation regarding the environment variable):

   ```bash
   $ HUGGINGFACE_CO_STAGING=1 pytest tests/<TEST_TO_RUN>.py
   ```
   
   > For the following commands leveraging the `make` utility, we recommend using the WSL system when running on
   > Windows. More information [here](https://docs.microsoft.com/en-us/windows/wsl/about).

   You can also run the full suite with the following command.

   ```bash
   $ make test
   ```

   `hugginface_hub` relies on `black` and `isort` to format its source code
   consistently. After you make changes, apply automatic style corrections and code verifications
   that can't be automated:

   ```bash
   $ make style
   ```

   `huggingface_hub` also uses `flake8` and a few custom scripts to check for coding mistakes. Quality
   control runs in CI, however you can also run the same checks with:

   ```bash
   $ make quality
   ```

   Once you're happy with your changes, add changed files using `git add` and
   make a commit with `git commit` to record your changes locally:

   ```bash
   $ git add modified_file.py
   $ git commit
   ```

   Please write [good commit messages](https://chris.beams.io/posts/git-commit/).

   It is a good idea to sync your copy of the code with the original
   repository regularly. This way you can quickly account for changes:

   ```bash
   $ git fetch upstream
   $ git rebase upstream/master
   ```

   Push the changes to your account using:

   ```bash
   $ git push -u origin a-descriptive-name-for-my-changes
   ```

6. Once you are satisfied (**and the checklist below is happy too**), go to the
   webpage of your fork on GitHub. Click on 'Pull request' to send your changes
   to the project maintainers for review.

7. It's ok if maintainers ask you for changes. It happens to core contributors
   too! So everyone can see the changes in the Pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.


### Checklist

1. The title of your pull request should be a summary of its contribution;
2. If your pull request addresses an issue, please mention the issue number in
   the pull request description to make sure they are linked (and people
   consulting the issue know you are working on it);
3. To indicate a work in progress please prefix the title with `[WIP]`, or mark 
   the PR as a draft PR. These are useful to avoid duplicated work, and to differentiate 
   it from PRs ready to be merged;
4. Make sure existing tests pass;
5. Add high-coverage tests. No quality testing = no merge.
6. Due to the rapidly growing repository, it is important to make sure that no files that would significantly weigh down the repository are added. This includes images, videos and other non-text files. We prefer to leverage a hf.co hosted `dataset` like
   the ones hosted on [`hf-internal-testing`](https://huggingface.co/hf-internal-testing) in which to place these files and reference
   them by URL. We recommend putting them in the following dataset: [huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images).
   If an external contribution, feel free to add the images to your PR and ask a Hugging Face member to migrate your images
   to this dataset.

### Tests

An extensive test suite is included to test the library behavior and several examples. Library tests can be found in
the [tests folder](https://github.com/huggingface/huggingface_hub/tree/main/tests).

The `huggingface_hub` library's normal behavior is to work with the production Hugging Face Hub. However,
for tests, we prefer to run on a staging version. In order to do this, it's important to set the 
`HUGGINGFACE_CO_STAGING` environment variable to `1` when running tests. It is preferred to pass this in when running the tests, than setting a permanent environmental variable, as shown below.

We use `pytest` in order to run the tests for the library . From the root of the
repository they can be run with the following:

```bash
$ HUGGINGFACE_CO_STAGING=1 python -m pytest -sv ./tests

In fact, that's how `make test` is implemented (sans the `pip install` line)!

You can specify a smaller set of tests in order to test only the feature
you're working on.

For example, the following will only run the tests hel in the `test_repository.py` file:

```bash
$ HUGGINGFACE_CO_STAGING=1 python -m pytest -sv ./tests/test_repository.py
```

And the following will only run the tests that include `tag` in their name:

```bash
$ HUGGINGFACE_CO_STAGING=1 python -m pytest -sv ./tests -k tag
```
