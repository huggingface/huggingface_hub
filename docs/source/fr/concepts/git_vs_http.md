<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Git ou HTTP?

`huggingface_hub` est une librairie qui permet d'interagir avec le Hugging Face Hub,
qui est une collection de dépots Git (modèles, datasets ou spaces).
Il y a deux manières principales pour accéder au Hub en utilisant `huggingface_hub`.

La première approche, basée sur Git, appelée approche "git-based", est rendue possible par la classe [`Repository`].
Cette méthode utilise un wrapper autour de la commande `git` avec des fonctionnalités supplémentaires conçues pour interagir avec le Hub. La deuxième option, appelée approche "HTTP-based" , consiste à faire des requêtes HTTP en utilisant le client [`HfApi`]. Examinons
les avantages et les inconvénients de ces deux méthodes.

## Repository: l'approche historique basée sur git

Initialement, `huggingface_hub` était principalement construite autour de la classe [`Repository`]. Elle fournit des
wrappers Python pour les commandes `git` usuelles, telles que `"git add"`, `"git commit"`, `"git push"`,
`"git tag"`, `"git checkout"`, etc.

Cette librairie permet aussi de gérer l'authentification et les fichiers volumineux, souvent présents dans les dépôts Git de machine learning. De plus, ses méthodes sont exécutables en arrière-plan, ce qui est utile pour upload des données durant l'entrainement d'un modèle.

L'avantage principal de l'approche [`Repository`] est qu'elle permet de garder une
copie en local du dépot Git sur votre machine. Cela peut aussi devenir un désavantage,
car cette copie locale doit être mise à jour et maintenue constamment. C'est une méthode
analogue au développement de logiciel classique où chaque développeur maintient sa propre copie locale
et push ses changements lorsqu'il travaille sur une nouvelle fonctionnalité.
Toutefois, dans le contexte du machine learning la taille des fichiers rend peu pertinente cette approche car
les utilisateurs ont parfois besoin d'avoir
uniquement les poids des modèles pour l'inférence ou de convertir ces poids d'un format à un autre sans avoir à cloner
tout le dépôt.

<Tip warning={true}>

[`Repository`] est maintenant obsolète et remplacée par les alternatives basées sur des requêtes HTTP. Étant donné son adoption massive par les utilisateurs,
la suppression complète de [`Repository`] ne sera faite que pour la version `v1.0`.

</Tip>

## HfApi: Un client HTTP plus flexible

La classe [`HfApi`] a été développée afin de fournir une alternative aux dépôts git locaux,
qui peuvent être encombrant à maintenir, en particulier pour des modèles ou datasets volumineux.
La classe [`HfApi`]  offre les mêmes fonctionnalités que les approches basées sur Git,
telles que le téléchargement et le push de fichiers ainsi que la création de branches et de tags, mais sans
avoir besoin d'un fichier local qui doit être constamment synchronisé.

En plus des fonctionnalités déjà fournies par `git`, La classe [`HfApi`] offre des fonctionnalités
additionnelles, telles que la capacité à gérer des dépôts, le téléchargement des fichiers
dans le cache (permettant une réutilisation), la recherche dans le Hub pour trouver
des dépôts et des métadonnées, l'accès aux fonctionnalités communautaires telles que, les discussions,
les pull requests et les commentaires.

## Quelle méthode utiliser et quand ?

En général, **l'approche HTTP est la méthode recommandée** pour utiliser `huggingface_hub`
[`HfApi`] permet de pull et push des changements, de travailler avec les pull requests, les tags et les branches, l'interaction avec les discussions
et bien plus encore. Depuis la version `0.16`, les méthodes HTTP-based peuvent aussi être exécutées en arrière-plan, ce qui constituait le
dernier gros avantage  de la classe [`Repository`].

Toutefois, certaines commandes restent indisponibles en utilisant [`HfApi`].
Peut être que certaines ne le seront jamais, mais nous essayons toujours de réduire le fossé entre ces deux approches.
Si votre cas d'usage n'est pas couvert, nous serions ravis de vous aider. Pour cela, ouvrez 
[une issue sur Github](https://github.com/huggingface/huggingface_hub)! Nous écoutons tous les retours nous permettant de construire
l'écosystème 🤗 avec les utilisateurs et pour les utilisateurs.

Cette préférence pour l'approche basée sur [`HfApi`] plutôt que [`Repository`] ne signifie pas que les dépôts stopperons d'être versionnés avec git sur le Hugging Face Hub. Il sera toujours possible d'utiliser les commandes `git` en local lorsque nécessaire.