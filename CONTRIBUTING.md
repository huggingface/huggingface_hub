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

Everyone is welcome to contribute, and we value everybody's contribution. Code is not the only way to help the community.
Answering questions, helping others, reaching out and improving the documentations are immensely valuable to the community.

It also helps us if you spread the word: reference the library from blog posts
on the awesome projects it made possible, shout out on Twitter every time it has
helped you, or simply star the repo to say "thank you".

Whichever way you choose to contribute, please be mindful to respect our
[code of conduct](https://github.com/huggingface/huggingface_hub/blob/main/CODE_OF_CONDUCT.md).

> Looking for a good first issue to work on?
> Please check out our contributing guide below and then select an issue from our [curated list](https://github.com/huggingface/huggingface_hub/contribute).
> Pick one and get started with it!

### The client library, `huggingface_hub`

This repository hosts the `huggingface_hub`, the client library that interfaces any Python script with the Hugging Face Hub.
Its implementation lives in `src/huggingface_hub`, while the tests are located in `tests/`.

There are many ways you can contribute to this client library:

- Fixing outstanding issues with the existing code;
- Contributing to the examples or to the documentation;
- Submitting issues related to bugs or desired new features.

## Submitting a new issue or feature request

Do your best to follow these guidelines when submitting an issue or a feature
request. It will make it easier for us to come back to you quickly and with good
feedback.

### Did you find a bug?

The `huggingface_hub` library is robust and reliable thanks to the users who notify us of
the problems they encounter. So thank you for reporting an issue.

First, we would really appreciate it if you could **make sure the bug was not
already reported** (use the search bar on Github under Issues).

Did not find it? :( So we can act quickly on it, please follow these steps:

- A short, self-contained, code snippet that allows us to reproduce the bug in less than 30s;
- Provide the _full_ traceback if an exception is raised by copying the text from your terminal in the issue description.
- Include information about your local setup. You can dump this information by running `hf env` in your terminal;

### Do you want a new feature?

A good feature request addresses the following points:

1. Motivation first:

- Is it related to a problem/frustration with the library? If so, please explain
  why and provide a code snippet that demonstrates the problem best.
- Is it related to something you would need for a project? We'd love to hear
  about it!
- Is it something you worked on and think could benefit the community?
  Awesome! Tell us what problem it solved for you.

2. Write a _full paragraph_ describing the feature;
3. Provide a **code snippet** that demonstrates its future use;
4. In case this is related to a paper, please attach a link;
5. Attach any additional information (drawings, screenshots, etc.) you think may help.

If your issue is well written, we're already 80% of the way there by the time you post it!

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

   Start by synchronizing your `main` branch with the `upstream/main` branch (more details in the [GitHub Docs](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork)):

   ```bash
   $ git checkout main
   $ git fetch upstream
   $ git merge upstream/main
   ```

   Once your `main` branch is synchronized, create a new branch from it:

   ```bash
   $ git checkout -b a-descriptive-name-for-my-changes
   ```

   **Do not** work on the `main` branch.

4. Set up a development environment by running the following command in a [virtual environment](https://docs.python.org/3/library/venv.html#creating-virtual-environments) or a conda environment you've created for working on this library:

   ```bash
   $ pip uninstall huggingface_hub # make sure huggingface_hub is not already installed
   $ pip install -e ".[dev]" # install in editable (-e) mode
   ```

5. Develop the features on your branch

6. Test your implementation!

   To make a good Pull Request you must test the features you have added.
   To do so, we use the `unittest` framework and run them using `pytest`:

   ```bash
   $ pytest tests -k <TEST_NAME>
   # or
   $ pytest tests/<TEST_FILE>.py
   ```

7. Format your code.

   `huggingface_hub` relies on [`ruff`](https://github.com/astral-sh/ruff) to format its source code consistently. You
   can apply automatic style corrections and code verifications with the following command:

   ```bash
   $ make style
   ```

   This command will update your code to comply with the standards of the `huggingface_hub` repository. A few custom
   scripts are also run to ensure consistency. Once automatic style corrections have been applied, you must test that
   it passes the quality checks:

   ```bash
   $ make quality
   ```

   Compared to `make style`, `make quality` will never update your code. In addition to the previous code formatter, it
   also runs [`ty`](https://docs.astral.sh/ty/) type checker to check for static typing issues. All those tests will also run
   in the CI once you open your PR but it is recommended to run them locally in order to iterate faster.

   > For the commands leveraging the `make` utility, we recommend using the WSL system when running on
   > Windows. More information [here](https://docs.microsoft.com/en-us/windows/wsl/about).

8. (optional) Alternatively, you can install pre-commit hooks so that these styles are applied and checked on files
   that you have touched in each commit:

   ```bash
   pip install pre-commit
   pre-commit install
   ```

   You only need to do the above once in your repository's environment. If for any reason you would like to disable
   pre-commit hooks on a commit, you can pass `-n` to your `git commit` command to temporarily disable pre-commit
   hooks.

   To permanently disable hooks, you can run the following command:

   ```bash
   pre-commit uninstall
   ```

9. Once you're happy with your changes, add changed files using `git add` and make a commit with `git commit` to record
   your changes locally:

   ```bash
   $ git add modified_file.py
   $ git commit
   ```

   Please write [good commit messages](https://chris.beams.io/posts/git-commit/).

   It is a good idea to sync your copy of the code with the original
   repository regularly. The following document covers it in length: [github documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork)

   And here's how you can do it quickly from your `git` commandline:

   ```bash
   $ git fetch upstream
   $ git rebase upstream/main
   ```

   Push the changes to your account using:

   ```bash
   $ git push -u origin a-descriptive-name-for-my-changes
   ```

10. Once you are satisfied (**and the [checklist below](https://github.com/huggingface/huggingface_hub/blob/main/CONTRIBUTING.md#checklist)
    is happy too**), go to the webpage of your fork on GitHub. Click on 'Pull request' to send your changes to the project maintainers for review.

11. It's ok if maintainers ask you for changes. It happens all the time to core contributors
    too! So everyone can see the changes in the Pull request, work in your local
    branch and push the changes to your fork. They will automatically appear in
    the pull request.

12. Once your changes have been approved, one of the project maintainers will
    merge your pull request for you. Good job!

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

We use `pytest` in order to run the tests for the library.
From the root of the repository they can be run with the following:

```bash
$ python -m pytest ./tests
```

You can specify a smaller set of tests in order to test only the feature you're working on.

For example, the following will only run the tests in the `test_repository.py` file:

```bash
$ python -m pytest ./tests/test_repository.py
```

And the following will only run the tests that include `tag` in their name:

```bash
$ python -m pytest ./tests -k tag
```
