<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Intégrez votre framework de ML avec le Hub

Le Hugging Face Hub facilite l'hébergement et le partage de modèles et de jeux de données.
Des [dizaines de librairies](https://huggingface.co/docs/hub/models-libraries) sont intégrées à cet écosystème. La communauté travaille constamment à en intégrer de nouvelles et contribue ainsi à faciliter la collaboration dans le milieu du machine learning. La librairie `huggingface_hub` joue un rôle clé dans ce processus puisqu'elle permet d'interagir avec le Hub depuis n'importe quel script Python.

Il existe quatre façons principales d'intégrer une bibliothèque au Hub :
1. **Push to Hub**  implémente une méthode pour upload un modèle sur le Hub. Cela inclut les paramètres du modèle, sa fiche descriptive (appelée [Model Card](https://huggingface.co/docs/huggingface_hub/how-to-model-cards)) et toute autre information pertinente liée au modèle (par exemple, les logs d'entraînement). Cette méthode est souvent appelée `push_to_hub()`.
2. **Download from Hub** implémente une méthode pour charger un modèle depuis le Hub. La méthode doit télécharger la configuration et les poids du modèle puis instancier celui-ci. Cette méthode est souvent appelée `from_pretrained` ou `load_from_hub()`.
3. **Widgets** affiche un widget sur la page d'accueil de votre modèle dans le Hub. Les widgets permettent aux utilisateurs de rapidement tester un modèle depuis le navigateur.

Dans ce guide, nous nous concentrerons sur les deux premiers sujets. Nous présenterons les deux approches principales que vous pouvez utiliser pour intégrer une librairie, avec leurs avantages et leurs inconvénients. Tout est résumé à la fin du guide pour vous aider à choisir entre les deux. Veuillez garder à l'esprit que ce ne sont que des conseils, et vous êtes libres de les adapter à votre cas d'usage.

Si les Widgets vous intéressent, vous pouvez suivre [ce guide](https://huggingface.co/docs/hub/models-adding-libraries#set-up-the-inference-api). Dans les deux cas, vous pouvez nous contacter si vous intégrez une librairie au Hub et que vous voulez être listé [dans la documentation officielle](https://huggingface.co/docs/hub/models-libraries).

## Une approche flexible: les helpers

La première approche pour intégrer une librairie au Hub est d'implémenter les méthodes `push_to_hub` et `from_pretrained` 
vous-même. Ceci vous donne une flexibilité totale sur le choix du fichier que vous voulez upload/download et sur comment
gérer les inputs spécifiques à votre framework. Vous pouvez vous référer aux guides : [upload des fichiers](./upload)
et [télécharger des fichiers](./download) pour en savoir plus sur la manière de faire. Par example, c'est de cette manière que l'intégration
de FastAI est implémentée (voir [`push_to_hub_fastai`] et [`from_pretrained_fastai`]).

L'implémentation peut varier entre différentes librairies, mais le workflow est souvent similaire.

### from_pretrained

Voici un exemple classique pour implémenter la méthode `from_pretrained`:

```python
def from_pretrained(model_id: str) -> MyModelClass:
   # Téléchargement des paramètres depuis le Hub
   cached_model = hf_hub_download(
      repo_id=repo_id,
      filename="model.pkl",
      library_name="fastai",
      library_version=get_fastai_version(),
   )

   # Instanciation du modèle
    return load_model(cached_model)
```

### push_to_hub

La méthode `push_to_hub` demande souvent un peu plus de complexité pour gérer la création du dépôt git, générer la model card et enregistrer les paramètres.
Une approche commune est de sauvegarder tous ces fichiers dans un dossier temporaire, les transférer sur le Hub avant de les supprimer localement.

```python
def push_to_hub(model: MyModelClass, repo_name: str) -> None:
   api = HfApi()

   # Créez d'un dépôt s'il n'existe pas encore, et obtenez le repo_id associé
   repo_id = api.create_repo(repo_name, exist_ok=True).repo_id

   # Sauvegardez tous les fichiers dans un chemin temporaire, et pushez les en un seul commit
   with TemporaryDirectory() as tmpdir:
      tmpdir = Path(tmpdir)

      # Sauvegardez les poids
      save_model(model, tmpdir / "model.safetensors")

      # Générez la model card
      card = generate_model_card(model)
      (tmpdir / "README.md").write_text(card)

      # Sauvegardez les logs
      # Sauvegardez le métriques d'évaluation
      # ...

      # Pushez vers le Hub
      return api.upload_folder(repo_id=repo_id, folder_path=tmpdir)
```

Ceci n'est qu'un exemple. Si vous êtes intéressés par des manipulations plus complexes (supprimer des fichiers distants,
upload des poids à la volée, maintenir les poids localement, etc.) consultez le guide [upload des fichiers](./upload).

### Limitations

Bien que très flexible, cette approche a quelques défauts, particulièrement en termes de maintenance. Les utilisateurs
d'Hugging Face sont habitués à utiliser certaines fonctionnalités lorsqu'ils travaillent avec `huggingface_hub`. Par exemple,
lors du chargement de fichiers depuis le Hub, il est commun de passer des paramètres tels que:
- `token`: pour télécharger depuis un dépôt privé
- `revision`: pour télécharger depuis une branche spécifique
- `cache_dir`: pour paramétrer la mise en cache des fichiers
- `force_download`: pour désactiver le cache
- `api_endpoint`/`proxies`: pour configurer la session HTTP

Lorsque vous pushez des modèles, des paramètres similaires sont utilisables:
- `commit_message`: message de commit personnalisé
- `private`: crée un dépôt privé s'il en manque un
- `create_pr`: crée une pull request au lieu de push vers `main`
- `branch`: push vers une branche au lieu de push sur `main`
- `allow_patterns`/`ignore_patterns`: filtre les fichiers à upload
- `token`
- `api_endpoint`
- ...

Tous ces paramètres peuvent être ajoutés aux implémentations vues ci-dessus et passés aux méthodes de `huggingface_hub`.
Toutefois, si un paramètre change ou qu'une nouvelle fonctionnalité est ajoutée, vous devrez mettre à jour votre package.
Supporter ces paramètres implique aussi plus de documentation à maintenir de votre côté. Dans la prochaine section, nous allons voir comment dépasser ces limitations.

## Une approche plus complexe: l'héritage de classe

Comme vu ci-dessus, deux méthodes principales sont à inclure dans votre librairie pour l'intégrer avec le Hub:
la méthode permettant d'upload des fichiers (`push_to_hub`) et celle pour télécharger des fichiers (`from_pretrained`).
Vous pouvez implémenter ces méthodes vous-même mais cela a des inconvénients. Pour gérer ça, `huggingface_hub` fournit
un outil qui utilise l'héritage de classe. Regardons comment ça marche !

Dans beaucoup de cas, une librairie définit déjà les modèles comme des classes Python. La classe contient les
propriétés du modèle et des méthodes pour charger, lancer, entraîner et évaluer le modèle. Notre approche est d'étendre
cette classe pour inclure les fonctionnalités upload et download en utilisant les mixins. Une [mixin](https://stackoverflow.com/a/547714)
est une classe qui est faite pour étendre une classe existante avec une liste de fonctionnalités spécifiques en utilisant l'héritage de classe.
`huggingface_hub` offre son propre mixin, le [`ModelHubMixin`]. La clef ici est de comprendre son comportement et comment le customiser.

La classe [`ModelHubMixin`] implémente 3 méthodes *public* (`push_to_hub`, `save_pretrained` et `from_pretrained`). Ce
sont les méthodes que vos utilisateurs appelleront pour charger/enregistrer des modèles avec votre librairie.
[`ModelHubMixin`] définit aussi 2 méthodes *private* (`_save_pretrained` et `from_pretrained`). Ce sont celles que vous
devez implémenter. Ainsi, pour intégrer votre librairie, vous devez :

1. Faire en sorte que votre classe Model hérite de [`ModelHubMixin`].
2. Implémenter les méthodes privées:
    - [`~ModelHubMixin._save_pretrained`]: méthode qui prend en entrée un chemin vers un directory et qui sauvegarde le modèle. 
    Vous devez écrire toute la logique pour dump votre modèle de cette manière: model card, poids du modèle, fichiers de configuration,
    et logs d'entraînement. Toute information pertinente pour ce modèle doit être gérée par cette méthode. Les
    [model cards](https://huggingface.co/docs/hub/model-cards) sont particulièrement importantes pour décrire votre modèle. Vérifiez
    [notre guide d'implémentation](./model-cards) pour plus de détails.
    - [`~ModelHubMixin._from_pretrained`]: **méthode de classe** prenant en entrée un `model_id` et qui retourne un modèle instancié.
    Cette méthode doit télécharger un ou plusieurs fichier(s) et le(s) charger.
3. Fini!

L'avantage d'utiliser [`ModelHubMixin`] est qu'une fois que vous vous êtes occupés de la sérialisation et du chargement du fichier,
vous êtes prêts. Vous n'avez pas besoin de vous soucier de la création du dépôt, des commits, des pull requests ou des révisions.
Tout ceci est géré par le mixin et est disponible pour vos utilisateurs. Le Mixin s'assure aussi que les méthodes publiques sont
bien documentées et que les annotations de typage sont spécifiées.

### Un exemple concret: PyTorch

Un bon exemple de ce que nous avons vu ci-dessus est [`PyTorchModelHubMixin`], notre intégration pour le framework PyTorch.
C'est une intégration prête à l'emploi.

#### Comment l'utiliser ?

Voici comment n'importe quel utilisateur peut charger/enregistrer un modèle Pytorch depuis/vers le Hub:

```python
>>> import torch
>>> import torch.nn as nn
>>> from huggingface_hub import PyTorchModelHubMixin

# 1. Définissez votre modèle Pytorch exactement comme vous êtes habitués à le faire
>>> class MyModel(nn.Module, PyTorchModelHubMixin): # héritage multiple
...     def __init__(self):
...         super().__init__() 
...         self.param = nn.Parameter(torch.rand(3, 4))
...         self.linear = nn.Linear(4, 5)

...     def forward(self, x):
...         return self.linear(x + self.param)
>>> model = MyModel()

# 2. (optionnel) Sauvegarder le modèle dans un chemin local
>>> model.save_pretrained("path/to/my-awesome-model")

# 3. Pushez les poids du modèle vers le Hub
>>> model.push_to_hub("my-awesome-model")

# 4. initialisez le modèle depuis le Hub
>>> model = MyModel.from_pretrained("username/my-awesome-model")
```

#### Implémentation

L'implémentation est très succincte (voir [ici](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hub_mixin.py)).

1. Premièrement, faites hériter votre classe de `ModelHubMixin`:

```python
from huggingface_hub import ModelHubMixin

class PyTorchModelHubMixin(ModelHubMixin):
   (...)
```

2. Implémentez la méthode `_save_pretrained`:

```py
from huggingface_hub import ModelCard, ModelCardData

class PyTorchModelHubMixin(ModelHubMixin):
   (...)

   def _save_pretrained(self, save_directory: Path):
      """Générez une model card et enregistrez les poids d'un modèle Pytroch vers un chemin local."""
      model_card = ModelCard.from_template(
         card_data=ModelCardData(
            license='mit',
            library_name="pytorch",
            ...
         ),
         model_summary=...,
         model_type=...,
         ...
      )
      (save_directory / "README.md").write_text(str(model))
      torch.save(obj=self.module.state_dict(), f=save_directory / "pytorch_model.bin")
```

3. Implémentez la méthode `_from_pretrained`:

```python
class PyTorchModelHubMixin(ModelHubMixin):
   (...)

   @classmethod # Doit absolument être une méthode de clase !
   def _from_pretrained(
      cls,
      *,
      model_id: str,
      revision: str,
      cache_dir: str,
      force_download: bool,
      local_files_only: bool,
      token: Union[str, bool, None],
      map_location: str = "cpu", # argument supplémentaire
      strict: bool = False, # argument supplémentaire
      **model_kwargs,
   ):
      """Chargez les poids pré-entrainés et renvoyez les au modèle chargé."""
      if os.path.isdir(model_id): # Peut être un chemin local
         print("Loading weights from local directory")
         model_file = os.path.join(model_id, "pytorch_model.bin")
      else: # Ou un modèle du Hub
         model_file = hf_hub_download( # Téléchargez depuis le Hub, en passant le mêmes arguments d'entrée
            repo_id=model_id,
            filename="pytorch_model.bin",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
         )

      # Chargez le modèle et reoutnez une logique personnalisée dépendant de votre framework
      model = cls(**model_kwargs)
      state_dict = torch.load(model_file, map_location=torch.device(map_location))
      model.load_state_dict(state_dict, strict=strict)
      model.eval()
      return model
```

Et c'est fini ! Votre librairie permet maintenant aux utilisateurs d'upload et de télécharger des fichiers vers et depuis le Hub.

## Comparaison

Résumons rapidement les deux approches que nous avons vu avec leurs avantages et leurs défauts. Le tableau ci-dessous
est purement indicatif. Votre framework aura peut-êre des spécificités à prendre en compte. Ce guide
est ici pour vous donner des indications et des idées sur comment gérer l'intégration. Dans tous les cas,
n'hésitez pas à nous contacter si vous avez une question !

<!-- Généré en utilisant https://www.tablesgenerator.com/markdown_tables -->
|            Intégration            |                                                                             Utilisant des helpers                                                                              |                                         Utilisant [`ModelHubMixin`]                                         |
| :-------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------: |
|      Expérience utilisateur       |                                                           `model = load_from_hub(...)`<br>`push_to_hub(model, ...)`                                                            |                     `model = MyModel.from_pretrained(...)`<br>`model.push_to_hub(...)`                      |
|             Flexible              |                                                        Très flexible.<br>Vous controllez complètement l'implémentation.                                                        |                     Moins flexible.<br>Votre framework doit avoir une classe de modèle.                     |
|            Maintenance            | Plus de maintenance pour ajouter du support pour la configuration, et de nouvelles fonctionnalités. Peut aussi nécessiter de fixx des problèmes signalés par les utilisateurs. | Moins de maintenance vu que la plupart des intégrations avec le Hub sont implémentés dans `huggingface_hub` |
| Documentation / Anotation de type |                                                                               A écrire à la main                                                                               |                                  Géré partiellement par `huggingface_hub`.                                  |
