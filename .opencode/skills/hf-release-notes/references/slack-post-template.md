# Slack post template

This is the template for the Slack announcement message. The script appends the "Pinging:" section
and closing line automatically — the skill only generates the body from the greeting through the
pip install command.

## Template

```
Hello @canal :hello: The next release of `huggingface_hub` (vX.Y.Z) is on its way! :tadaco:

Release notes :point_right: https://github.com/huggingface/huggingface_hub/releases/tag/vX.Y.Z

:sparkles: Highlights
 :emoji: Feature name: 1-2 sentence summary of the feature.
 :emoji: Another feature: brief description.
 A bunch of QoL improvements to the CLI:
  Sub-feature 1
  Sub-feature 2

<If breaking changes>
:warning: Breaking changes: brief description of what changed.
<If no breaking changes>
No breaking changes in this release.

We also introduced a bunch of QoL improvements and fixes!

You can try the pre-release now:
pip install -U huggingface_hub==X.Y.ZrcN
```

## Real examples

### Example 1 (v1.7.0)

```
Hello @canal :hello: The next release of huggingface_hub (v1.7.0) is on its way! :tadaco:

Release notes :point_right: https://github.com/huggingface/huggingface_hub/releases/tag/v1.7.0

:sparkles: Highlights
 :package: CLI Extensions : extensions can now be full pip-installable Python packages (this may have slipped into the v1.6.0 release notes but is definitely shipped here :see_no_evil:). Plus a new hf extensions search command to discover extensions from the terminal.
hf-xet bumped to 1.4.2 with upload optimizations and a fix for deadlocks on large file downloads. that should improve upload speed for large files
 A bunch of QoL improvements to the CLI:
  All list commands normalized to list / ls aliases
  Hidden --json shorthand for --format json
  num_parameters filtering in hf models list
Allow hf skills add to default to installing from the skills directory
hf auth login has a new flag --force to override  if already logged in

No breaking changes in this release.

You can try the pre-release now:
pip install -U huggingface_hub==1.7.0rc1
```

### Example 2 (v1.5.0)

```
Hello @canal :hello: The next release of `huggingface_hub` (v1.5.0) is on its way! :tadaco:

Release notes :point_right: https://github.com/huggingface/huggingface_hub/releases/tag/v1.5.0.rc0

:sparkles: Highlights
:bucket: Buckets API! Create, list, sync, and manage buckets on the Hub. Sync local directories with hf buckets sync ./data hf://buckets/username/my-bucket.
:robot_face: AI-first CLI: hf skills add to install skills for Claude, Codex, etc., output options (json/table) for composability, aliases for better consistency, etc.
:fire: Spaces hot-reload: Patch Python files in a running Space with hf spaces hot-reload username/repo-name app.py.
:electric_plug: CLI Extensions: Install and run external CLI extensions from GitHub repos with `hf extensions install hf-claude`.
:chart_with_upwards_trend: Jobs: Multi-GPU training support, hf jobs hardware command, better filtering with labels, and --follow/-f logs flag.

:warning: Breaking changes: hf repo is deprecated in favor of `hf repos`. Also, hf repo-files delete has moved to hf repo delete-files. Aliases are kept for backward compatibility.

We also introduced a bunch of QoL improvements and fixes!

You can try the pre-release now:
pip install huggingface_hub==1.5.0rc0
```

### Example 3 (v1.6.0)

```
Hello @canal :hello: The next release of huggingface_hub (v1.6.0) is on its way! :tadaco:

Release notes :point_right: https://github.com/huggingface/huggingface_hub/releases/tag/v1.6.0

:sparkles: Highlights
• :bucket: HfFileSystem for Buckets! Reading from buckets is as simple as pd.read_csv("hf://buckets/.../affluence.csv")  (thanks to @Quentin Lhoest).
• :computer: New CLI commands:
 hf spaces dev-mode to enable dev mode and get SSH/VSCode connection instructions (cc @Quentin Lhoest)
 hf discussions for full discussion & PR management
hf webhooks for CRUD on Hub webhooks
hf datasets parquet + hf datasets sql to discover and query dataset parquet files with DuckDB cc @Caleb Fahlgren
hf repos duplicate to duplicate any repo.
• :electric_plug: pip-installable CLI extensions: hf extensions install now supports Python packages in addition to executables
• :zap: NVIDIA provider support added to InferenceClient.

:warning: Breaking change: the deprecated direction argument in list_models, list_datasets and list_spaces has been removed. (was announced since months)

We also introduced a bunch of QoL improvements and fixes!

You can try the pre-release now:
pip install -U huggingface_hub==1.6.0rc0
```
