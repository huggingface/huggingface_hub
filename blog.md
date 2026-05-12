# From six-week releases to weekly: how we automated `huggingface_hub` for $0.30 a release

`huggingface_hub` is the Python client at the base of the Hugging Face ecosystem. It's used by `transformers`, `datasets`, `diffusers`, and dozens of other libraries to talk to the Hub, so every delay in releasing it is a delay downstream.

For a long time, we released it every 4 to 6 weeks. We now release it every week, through a single GitHub Actions workflow that costs less than $0.30 per release to run. This post walks through what changed, how the workflow is put together, and why we think this pattern will become more common as open-weights models get cheaper and more capable.

## Where we started

Our previous release process was **half automated, half manual**.

Already automated in CI:

- Publishing to PyPI once a tag was pushed.
- Opening test branches in downstream libraries (`transformers`, `datasets`, `diffusers`, `sentence-transformers`) with the RC version pinned.

Still manual, every time:

- Creating the release branch locally, updating the version in `__init__.py`, committing, tagging, pushing.
- Checking the downstream CI runs and triaging failures.
- Scrolling through every PR merged since the last release and writing release notes by hand — grouped by theme, with context, in a tone that didn't read like a `git log` dump.
- Cutting the stable release after the RC period.
- Writing an internal Slack announcement for the team.
- Drafting LinkedIn and X posts.
- Opening the post-release PR to bump `main` to the next `dev0` version.

Writing good release notes for a minor version was the heaviest part. Thirty PRs, three or four themes, some user-facing and some internal — it's the kind of work that's not technically difficult but needs a few hours of focus. Multiply that by every announcement we had to draft afterwards and releasing a minor version was easily a half-day of work spread over several days.

One thing we *didn't* do in the old process: comment on each PR once it shipped. There was no good way to fit it in, so contributors (and we ourselves) had no easy way to tell when a given PR had been released. This turned out to matter more than we expected, as described below.

## What the workflow does today

The full workflow is a single file, [`.github/workflows/release.yml`](https://github.com/huggingface/huggingface_hub/blob/main/.github/workflows/release.yml), triggered manually from the GitHub Actions UI. It takes one input: a release type, chosen from `minor-prerelease`, `minor-release`, or `patch-release`.

The jobs run roughly in this order:

- **Prepare**. Compute the next version number, create or reuse the release branch, update `__version__`, commit, tag, push. For a minor pre-release this also creates the release branch from `main`; for a stable or patch release it reuses the existing one.
- **Publish to PyPI**. Build and upload the `huggingface_hub` package. In parallel, build and upload the `hf` CLI package to its own PyPI distribution.
- **Generate release notes**. Diff the commit range against the previous tag, pull PR metadata from the GitHub API, and use a language model to produce a structured changelog. The result is saved as a draft GitHub release.
- **Open downstream test branches**. For pre-releases, open a branch in each downstream repo with the RC version pinned, so their CI tells us quickly whether anything broke.
- **Generate the Slack announcement**. Re-read the release notes (possibly after a human has edited them) and produce an internal announcement in the voice we use for our team channel.
- **Generate social media drafts**. Produce LinkedIn- and X-ready drafts and upload them to a Hugging Face Bucket so the marketing team can pick them up.
- **Post-release bump**. After a stable release, open a PR on `main` to bump to the next `dev0` version.
- **Comment on shipped PRs**. Walk through each PR merged into this release and leave a comment linking to the release it shipped in.
- **Sync CLI documentation**. Push the latest CLI skill docs to our `skills` repo.
- **Report to Slack**. Each step posts its status as a thread reply on a release message; the final step updates the root message with ✅ or ❌.

From a human perspective, triggering a release is four clicks in the Actions UI. The only remaining manual step is editing and publishing the draft GitHub release once a human has reviewed the generated notes.

## The AI-powered parts

Three jobs use a language model: release notes, Slack announcement, and social media drafts. All three use [OpenCode](https://opencode.ai/) as the agent runtime and [GLM 5.1](https://huggingface.co/zai-org/GLM-4.5) (configurable through a repo variable) served via [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index).

The release notes job is the one we tuned the most. The input is a prompt plus the list of PRs merged since the last stable release, with titles and bodies. The output is a Markdown file grouped by theme — new features, improvements, fixes, internal changes — with callouts for anything user-facing. A human reviewer edits the draft before publishing, but the first pass is usually close enough that the edits are light.

The Slack and social post generators are smaller. They read the release notes (edited version included) and produce the right voice for each channel: detailed for internal Slack, concise for LinkedIn and X. The prompts are short; the models are good at this kind of focused rewriting.

All three jobs are standard Python scripts that shell out to OpenCode. Nothing exotic. If the inference call fails or produces empty output, the job fails loudly rather than publishing an empty release.

## Cost

A full release — notes, Slack draft, and social posts — currently costs us **less than $0.30** on Hugging Face Inference Providers. That covers several rounds of prompting across the three AI-powered jobs, on a release that typically spans 20 to 40 PRs.

Every piece of the stack is something any maintainer can use:

- GitHub Actions: free on public repos.
- OpenCode: open source.
- GLM 5.1: open weights, served by multiple Inference Providers.
- Hugging Face Inference Providers: pay-as-you-go, no minimum.

No enterprise contract, no platform lock-in. The YAML file is ~1200 lines and has been through about 55 commits of iteration — most of it is plain shell, and the parts that aren't generic to `huggingface_hub` are easy to spot and adapt.

## What changed in practice

Our release cadence went from roughly one release every 4 to 6 weeks to one per week, consistently. The switch wasn't a one-off effort — it's the natural consequence of lowering the cost of each release. When a release takes fifteen minutes of human attention instead of half a day, you ship whenever there is something worth shipping.

A few secondary effects were more interesting than the cadence itself:

- **Release notes got better, not worse.** The first draft always being there means the reviewer's time goes into polishing rather than writing from a blank page. We end up with more consistent grouping and fewer omissions.
- **Downstream breakages surface earlier.** Because the test branches in `transformers` et al. are opened automatically on every pre-release, integration issues show up during the RC phase, not after the stable release is out.
- **Contributor feedback loops shortened.** The automatic "this shipped in v1.13.0" comment on each merged PR is the addition we originally thought was a nice-to-have. It has turned out to be useful in a concrete way: when a user reports an issue on a closed PR, we (and they) can immediately see which release the fix is in. That context used to require a manual git and tag lookup.
- **The on-call cost of releases dropped to near zero.** If the usual release person is away, anyone on the team can click the button.

## What's next

A few directions we're looking at:

- **Triaging downstream failures automatically.** Today the workflow opens test branches; a human reads the CI results. An obvious next step is a job that reads the failing logs and produces a short summary of whether the break is on our side, theirs, or a test flake.
- **Better structure in release notes.** When a release spans 40 PRs across many areas, even a strong model can flatten the narrative. We'd like to experiment with a two-pass approach: cluster the PRs first, then write each section against its cluster.
- **Extending the pattern to other libraries.** The workflow is shaped around `huggingface_hub`, but most of the structure is generic. We expect to reuse large parts of it across other Python libraries in the ecosystem.

## Takeaway

The parts of a release that used to require a half-day of focused human work — writing release notes, drafting announcements, coordinating downstream checks — are the parts language models are good at drafting. Everything else is mechanical and fits comfortably in a YAML file.

We had no special infrastructure to build. The ingredients were already open and accessible: GitHub Actions, an open-source agent runtime, open-weights models, and a pay-as-you-go inference API. Stitching them together took iteration, but no large investment. For less than a dollar a release, we've moved from a multi-week cadence to a weekly one without adding process debt.

If you maintain a Python library and recognize the pattern — half-automated, half-manual, release day eats your afternoon — the full workflow file is public. Fork it, adapt it, and let us know how it goes.
