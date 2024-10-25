<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Managing your Space runtime

Check the [`HfApi`] documentation page for the reference of methods to manage your Space on the Hub.

- Duplicate a Space: [`duplicate_space`]
- Fetch current runtime: [`get_space_runtime`]
- Manage secrets: [`add_space_secret`] and [`delete_space_secret`]
- Manage hardware: [`request_space_hardware`]
- Manage state: [`pause_space`], [`restart_space`], [`set_space_sleep_time`]

## Data structures

### SpaceRuntime

[[autodoc]] SpaceRuntime

### SpaceHardware

[[autodoc]] SpaceHardware

### SpaceStage

[[autodoc]] SpaceStage

### SpaceStorage

[[autodoc]] SpaceStorage

### SpaceVariable

[[autodoc]] SpaceVariable
