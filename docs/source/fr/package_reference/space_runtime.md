<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Gérez le temps d'exécution de votre space

Consultez la page de documentation d'[`HfApi`] pour les références des méthodes pour gérer votre space
sur le Hub.

- Dupliquer un space: [`duplicate_space`]
- Afficher les temps de calcul actuels: [`get_space_runtime`]
- Gérer les secrets: [`add_space_secret`] et [`delete_space_secret`]
- Gérer le hardware: [`request_space_hardware`]
- Gérer l'état: [`pause_space`], [`restart_space`], [`set_space_sleep_time`]

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


