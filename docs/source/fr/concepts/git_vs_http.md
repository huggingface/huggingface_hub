<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Paradigme Git vs HTTP

La bibliothèque `huggingface_hub` est une bibliothèque qui permet d'interagir avec le Hugging Face Hub, une collection de dépôts basés sur Git (modèles, jeux de données ou Spaces). Il existe deux façons principales d'accéder au Hub en utilisant `huggingface_hub`.

La première approche, dite "basée sur Git", repose sur l'utilisation de commandes `git` standard directement dans un terminal. Cette méthode vous permet de cloner des dépôts, créer des commits et pousser des modifications manuellement. La seconde option, appelée approche "basée sur HTTP", consiste à effectuer des requêtes HTTP en utilisant le client [`HfApi`]. Examinons les avantages et inconvénients de chaque approche.

## Git : l'approche CLI historique

Au début, la plupart des utilisateurs interagissaient avec le Hugging Face Hub en utilisant de simples commandes `git` telles que `git clone`, `git add`, `git commit`, `git push`, `git tag` ou `git checkout`.

Cette approche vous permet de travailler avec une copie locale complète du dépôt sur votre machine, exactement comme dans le développement logiciel traditionnel. Cela peut être un avantage lorsque vous avez besoin d'un accès hors ligne ou que vous souhaitez travailler avec l'historique complet d'un dépôt. Cependant, cela comporte également des inconvénients : vous êtes responsable de maintenir le dépôt à jour localement, de gérer les identifiants et de gérer les fichiers volumineux (via `git-lfs`), ce qui peut devenir fastidieux lorsque vous travaillez avec de grands modèles de machine learning ou datasets.

Dans de nombreux workflows de machine learning, vous n'avez peut-être besoin que de télécharger quelques fichiers pour l'inférence ou convertir des poids sans avoir besoin de cloner l'ensemble du dépôt. Dans de tels cas, utiliser `git` peut être excessif et introduire une complexité inutile. C'est pourquoi nous avons développé une alternative basée sur HTTP.

## HfApi : un client HTTP flexible et pratique

La classe [`HfApi`] a été développée pour offrir une alternative à l'utilisation de dépôts Git locaux, qui peuvent être fastidieux à maintenir comme dit précédemment. La classe [`HfApi`] offre les mêmes fonctionnalités que les workflows basés sur Git - comme télécharger et pousser des fichiers, créer des branches et des tags - mais sans avoir besoin d'un dossier local qui doit être maintenu synchronisé.

En plus des fonctionnalités déjà fournies par `git`, la classe [`HfApi`] offre des fonctionnalités supplémentaires, telles que la possibilité de gérer des dépôts, de télécharger des fichiers en utilisant la mise en cache pour une réutilisation efficace, de rechercher dans le Hub des dépôts et métadonnées. Elle permet aussi d'accéder aux fonctionnalités communautaires telles que les discussions, PRs et commentaires, la configuration du matériel et les secrets des Spaces.

## Que dois-je utiliser ? Et quand ?

Dans l'ensemble, **l'approche basée sur HTTP est la méthode recommandée pour utiliser** `huggingface_hub` dans tous les cas. [`HfApi`] vous permet de récupérer et pousser des modifications, travailler avec des PRs, tags et branches, et interagir avec les discussions et bien plus encore.

Cependant, toutes les commandes Git ne sont pas disponibles via [`HfApi`]. Certaines ne seront peut-être jamais implémentées, mais nous essayons toujours d'améliorer et de combler cette écart. Si vous ne voyez pas votre cas d'usage couvert, veuillez ouvrir [une issue sur GitHub](https://github.com/huggingface/huggingface_hub) ! Nous apprécions les retours !

Cette préférence pour l'approche HTTP basée sur [`HfApi`] plutôt que les commandes `git` directes ne signifie pas que le versionnement Git disparaîtra du Hugging Face Hub de sitôt. Il sera toujours possible d'utiliser `git` localement dans les workflows où cela a du sens.
