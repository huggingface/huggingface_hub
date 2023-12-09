<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Paradigme Git vs HTTP

La librairie `huggingface_hub` est une librairie qui permet d'intéragir avec le Hub Hugging Face,
qui est une collection de dépots Git (modèles, datasets ou espaces).
Il y a deux manières principales pour accéder au Hub en utilisant `huggingface_hub`.

La première approche, basée sur Git, appelée approche "git-based", est rendue possible par la classe [`Repository`].
Cette méthode utilise un wrapper autour de la commande `git` avec des fonctionnalités supplémentaires conçues pour intéragir avec le Hub. La deuxième option, appelée approche "HTTP-based" ,
nécessite de faire des requêtes HTTP en utilisant le client [`HfApi`]. Éxaminions
les avantages et les inconvénients de ces deux méthodes.

## Repository: L'approche hstorique basée sur git

Au début, `huggingface_hub` était principalement construit autour de la classe [`Repository`]. Elle fournit des
wrappers Python pour les commandes `git` usuelles, telles que `"git add"`, `"git commit"`, `"git push"`,
`"git tag"`, `"git checkout"`, etc.

Cette librairie permet aussi de définir les données d'identification et de suivre les fichiers volumineux, qui sont souvent utilisés dans les dépot Git de machine learning. De plus, la librairie vous permet d'exécuter ses
méthodes en arrière-plan, ce qui la rend utile pour upload des données pendant l'entrainement des modèles.

L'avantage principal de l'utilisation de [`Repository`] est que cette méthode permet de garder une
copie en local de tout le dépot Git sur votre machine. Cela peut aussi devenir un désavantage,
car cette copie locale doit être mise à jour et maintenue constamment. C'est une manière de procéder
analogue au développement de logiciel traditionnel où chaque développeur maintient sa propre copie locale
et push les changement lorsqu'ils travaillent sur une fonctionnalité.
Toutefois, dans le contexte du machine learning,
elle n'est pas toujours pertinente car les utilisateurs ont parfois uniquement besoin d'avoir
les poids des modèles pour l'inférence ou de convertir ces poids d'un format à un autre sans avoir à cloner
tout le dépôt.

<Tip warning={true}>

[`Repository`] est maintenant deprecated et remplacé par les alternatives basées sur l'HTTP. Étant donné son adoption massive par les utilisateurs,
la suppression complète de [`Repository`] ne sera faite que pour la version `v1.0`.

</Tip>

## HfApi: Un client HTTP flexible et pratique

La classe [`HfApi`] a été développée afin de fournir une alternative aux dépôts git locaux,
qui peuvent être encombrant à maintenir, en particulier lors de l'utilisation de gros modèles ou de datasets volumineux.
La classe [`HfApi`]  offre les mêmes fonctionnalités que les approches basées sur Git,
telles que le téléchargement et le push de fichier ainsi que la création de branches et de tags, mais sans
avoir besoin d'un fichier local qui doit être constamment synchronisé.

En plus des fonctionnalités déjà fournies par `git`, La classe [`HfApi`] offre des fonctionnalités
additionnelles, telles que la capacité de gérer des dépots, le téléchargement des fichiers
en utilisant le cache pour une réutilisation plus efficace, la recherche dans le Hub pour trouver
des dépôts et des métadonnées, l'accès aux fonctionnalités de communautés telles que, les dicussions,
les pull requests, les commentaires, et la configuration d'espaces hardwares et de secrets.

## Quelle méthode utiliser et quand ?

En général, **L'approche basée sur l'HTTP est la méthode recommandée** pour l'utilisation d'`huggingface_hub`.
[`HfApi`] permet de pull et push des changements, travailler avec les pull requests, les tags et les branches, l'intréaction avec les discussions
et bien plus encore. Depuis la sortie  `0.16`, les méthodes basées sur l'HTTP peuvent aussi tourner en arrière plan, ce qui était le
dernier gros avantage  de la classe [`Repository`] sur [`HfApi`].

Toutefois, certaines commandes restent indisponibles en utilisant [`HfApi`].
Peut être que certaines ne le seront jamais, mais nous essayons toujours de réduire le fossé entre les deux approches.
Si votre cas d'usage n'est pas couvert, nous serions ravis de vous aider. Pour cela, ouvrez 
[une issue sur Github](https://github.com/huggingface/huggingface_hub)! Nous sommes prêt à entendre tout type de retour nous permettant de construire
l'écosystème 🤗 avec les utilisateurs et pour les utilisateurs.

Cette préférence pour l'approche basé sur [`HfApi`] au détriment de celle basée sur [`Repository`] ne signifie pas que le versioning git disparaitra
du Hub Hugging Face. Il sera toujours possible d'utiliser les commandes `git` en local lorsque cela a du sens.