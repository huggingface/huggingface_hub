<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# TensorBoard logger

TensorBoard est un toolkit de visualisation pour l'expérimentation en machine learning. TensorBoard permet de suivre et visualiser
des métriques telles que la perte et la précision, visualiser le graphe du modèle, afficher des histogrammes, afficher des images et bien plus encore.
TensorBoard est bien intégré avec le Hugging Face Hub. Le Hub détecte automatiquement les traces TensorBoard (telles que
`tfevents`) lorsqu'elles sont poussées vers le Hub, ce qui démarre une instance pour les visualiser. Pour obtenir plus d'informations sur l'intégration
TensorBoard sur le Hub, consultez [ce guide](https://huggingface.co/docs/hub/tensorboard).

Pour bénéficier de cette intégration, `huggingface_hub` fournit un logger personnalisé pour pousser les logs vers le Hub. Il fonctionne comme un
remplacement direct pour [SummaryWriter](https://tensorboardx.readthedocs.io/en/latest/tensorboard.html) sans code supplémentaire
nécessaire. Les traces sont toujours sauvegardées localement et un job en arrière-plan les pousse vers le Hub à intervalles réguliers.

## HFSummaryWriter

[[autodoc]] HFSummaryWriter
