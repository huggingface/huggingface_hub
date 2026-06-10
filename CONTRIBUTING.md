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
Answering questions, helping others, reaching out and improving the documentation are immensely valuable to the community.

It also helps us if you spread the word: reference the library from blog posts
on the awesome projects it made possible, shout out on social media every time it has
helped you, or simply star the repo to say "thank you".

Whichever way you choose to contribute, please be mindful to respect our
[code of conduct](https://github.com/huggingface/huggingface_hub/blob/main/CODE_OF_CONDUCT.md).

## Found a bug or want a new feature? Open an issue first

If you want to report a bug or suggest a technical improvement, **please open an issue rather than a pull request**.
A well-written issue with concrete details (*what* is wrong or missing, *why* it matters, *how* you'd expect it to work)
is the most helpful thing you can do. It helps us prioritize, design the right solution, and ship a fix quickly.

In practice, **we prefer to implement most code changes ourselves**. As maintainers we can iterate fast, keep the code
consistent, and ship fixes in a single pass. We will typically pick up issues and open a PR on our side when the timing
is right. This is not about gatekeeping. It is simply how we work most efficiently on a fast-moving codebase, and it
avoids long back-and-forth review cycles that end up being frustrating for everyone.

Pull requests are still welcome for documentation improvements, typo fixes, or changes that have been discussed and
scoped in an issue beforehand. But when in doubt, start with an issue.

If you open a pull request, you are expected to understand the code you submit. Using AI to help write code is fine. Submitting AI-generated slop you cannot explain is not.

If you use an agent, run it from the `huggingface_hub` root directory so it automatically picks up `AGENTS.md`. Your agent must follow the rules and guidelines defined there.

### The client library, `huggingface_hub`

This repository hosts the `huggingface_hub`, the client library that interfaces any Python script with the Hugging Face Hub.
Its implementation lives in `src/huggingface_hub`, while the tests are located in `tests/`.

## Submitting a new issue or feature request

Do your best to follow these guidelines when submitting an issue or a feature
request. It will make it easier for us to come back to you quickly and with good
feedback.

### Did you find a bug?

The `huggingface_hub` library is robust and reliable thanks to the users who notify us of
the problems they encounter. So thank you for reporting an issue.

First, we would really appreciate it if you could **make sure the bug was not
already reported** (use the search bar on GitHub under Issues).

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
   $ git clone git@github.com:<your GitHub handle>/huggingface_hub.git
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

4. Set up a development environment. We recommend using [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
   for a fast and reliable setup:

   ```bash
   $ uv venv .venv
   $ source .venv/bin/activate
   $ uv pip install -e ".[dev]"
   ```

   > **Windows users:** we recommend using [WSL](https://docs.microsoft.com/en-us/windows/wsl/about) for development.
   > The `make` commands below require a Unix-like shell.

5. Develop the features on your branch.

6. Format your code and check quality. Always run both before committing:

   ```bash
   $ make style   # auto-format code + regenerate generated files
   $ make quality  # check formatting, linting, and type errors (read-only)
   ```

   `make style` applies [`ruff`](https://github.com/astral-sh/ruff) formatting and updates auto-generated files
   (static imports, async client, CLI docs). `make quality` runs the same checks without modifying files and
   additionally runs the [`ty`](https://docs.astral.sh/ty/) type checker. Both will run in CI, but running them
   locally lets you iterate faster.

7. Test your implementation. You must test the features you have added:

   ```bash
   $ pytest tests/<TEST_FILE>.py           # run a specific test file
   $ pytest tests -k <TEST_NAME>           # run tests matching a name
   ```

8. Once you're happy with your changes, commit and push:

   ```bash
   $ git add modified_file.py
   $ git commit
   $ git push -u origin a-descriptive-name-for-my-changes
   ```

   Please write [good commit messages](https://chris.beams.io/posts/git-commit/).

   Keep your branch up to date with upstream:

   ```bash
   $ git fetch upstream
   $ git rebase upstream/main
   ```

9. Once you are satisfied (**and the [checklist below](https://github.com/huggingface/huggingface_hub/blob/main/CONTRIBUTING.md#checklist)
    is happy too**), go to the webpage of your fork on GitHub. Click on 'Pull request' to send your changes to the project maintainers for review.

10. It's ok if maintainers ask you for changes. It happens all the time to core contributors
    too! So everyone can see the changes in the Pull request, work in your local
    branch and push the changes to your fork. They will automatically appear in
    the pull request.

11. Once your changes have been approved, one of the project maintainers will
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

We use `pytest` to run the tests. From the root of the repository:

```bash
$ pytest tests/                          # run all tests (slow, many require network)
$ pytest tests/test_repository.py        # run a specific test file
$ pytest tests -k tag                    # run tests matching a name
```

#### Xet vs non-Xet tests

Whether a test depends on Xet is declared explicitly with markers, enforced by the
`xet_mode` fixture in `tests/conftest.py`:

- `@pytest.mark.xet` — the test **requires** `hf_xet` (e.g. Buckets, Xet upload/download).
  It is skipped when `hf_xet` is not installed and runs with Xet force-enabled otherwise.
  Mark a whole module with `pytestmark = pytest.mark.xet`.
- `@pytest.mark.no_xet` — the test **must run without** Xet (e.g. legacy LFS behavior).
  Xet is force-disabled for it, even if `hf_xet` is installed.
- unmarked — the test must work **regardless of Xet**. Nothing is forced: it runs with
  whatever your environment provides. In CI, unmarked tests run twice: once with
  `hf_xet` installed ("Xet only" job) and once without (other jobs). If an unmarked
  test only passes in one mode, mark it `xet` or `no_xet` accordingly.

```bash
$ pytest tests -m xet                    # only Xet-required tests (needs hf_xet)
$ pytest tests -m no_xet                 # only legacy (Xet force-disabled) tests
$ pytest tests -m "not xet"              # what CI runs without hf_xet installed
$ pytest tests -m "not no_xet"           # what CI runs with hf_xet installed
```
