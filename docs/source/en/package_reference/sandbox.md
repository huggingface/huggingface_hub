<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Sandboxes

> [!NOTE]
> The Sandbox API is experimental. Its API and behavior may change without notice. Shared sandboxes are intended for
> workloads within the same trust boundary; use dedicated sandboxes for workloads that do not trust each other.

Check out the [Sandboxes guide](../guides/sandbox) to learn how to use them.

## Sandbox

[[autodoc]] Sandbox

## SandboxPool

[[autodoc]] SandboxPool

## Data structures

### SandboxCommandResult

[[autodoc]] SandboxCommandResult

### SandboxProcess

[[autodoc]] SandboxProcess

### FileEntry

[[autodoc]] huggingface_hub._sandbox.FileEntry

## Errors

### SandboxError

[[autodoc]] huggingface_hub.errors.SandboxError

### SandboxCommandError

[[autodoc]] huggingface_hub.errors.SandboxCommandError
