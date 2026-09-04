<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Gérer le runtime de votre Space

Consultez la page de documentation [`HfApi`] pour la référence des méthodes permettant de gérer votre Space sur le Hub.

- Dupliquer un Space : [`duplicate_space`]
- Récupérer le runtime actuel : [`get_space_runtime`]
- Gérer les secrets : [`add_space_secret`] et [`delete_space_secret`]
- Gérer le hardware : [`request_space_hardware`]
- Gérer l'état : [`pause_space`], [`restart_space`], [`set_space_sleep_time`]

## Structures de données

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
