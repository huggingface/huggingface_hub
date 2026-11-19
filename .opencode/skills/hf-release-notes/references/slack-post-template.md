# Slack post template

This is the template for the Slack announcement message. The script appends the "Ping:" section
and closing line automatically — the skill only generates the body from the greeting through the
breaking changes line.

## Template

```
Hello @channel :hello: Release `huggingface_hub vX.Y.Z` is on its way!

Release notes :point_right: https://github.com/huggingface/huggingface_hub/releases/tag/vX.Y.Z

**Highlights:**
- **[Feature name]** 1 sentence summary of the feature.
- **[Another feature]** brief description.

<If breaking changes>
:warning: Breaking changes: brief description of what changed.
<If no breaking changes>
No breaking changes in this release.
```
