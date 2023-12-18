<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Logger TensorBoard

TensorBoard est un kit d'outils pour l'expérimentation avec les outils de machine learning. TensorBoard permet de traquer
et de visualiser les métriques telles que la fonction de perte la précision, visualiser le graphe, visualisers des
histogrammes, afficher des images et bien plus. TensorBoard est bien intégré avec le Hub Hugging Face. Le Hub détecte
automatiquement les traces de Tensorboard (telles que `tfevents`) lors d'un push vers le Hub qui lance une instance
pour les visualiser. Pour avoir plus d'informations sur l'intégration de TensorBoard avec le Hub, consultez [ce guide](https://huggingface.co/docs/hub/tensorboard).

Pour bénéficier de cette intégration, `huggingface_hub` fournit un logger personnalisé pour push les logs vers le Hub.
Il fonctionne comme un remplacement de [SummaryWriter](https://tensorboardx.readthedocs.io/en/latest/tensorboard.html)
avec aucun code supplémentaire nécessaire. Les traces sont toujours enregistrées en local et sont push vers le Hub
à des intervalles réguliers en arrière plan.

## HFSummaryWriter

[[autodoc]] HFSummaryWriter