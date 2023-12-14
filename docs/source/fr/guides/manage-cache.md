<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Gérer le cache-system `huggingface_hub`

## Comprendre le caching

Le cache-system Hugging Face Hub a été créé pour être le cache central partagé par toutes les
librairies dépendant du Hub. Il a été mis à jour dans la version v0.8.0 pour éviter de
retélécharger les mêmes fichiers entre chaque révisions.

Le système de cache fonctionne comme suit:

```
<CACHE_DIR>
├─ <MODELS>
├─ <DATASETS>
├─ <SPACES>
```

Le `<CACHE_DIR>` est souvent votre chemin vers la home de votre utilisateur. Cependant, vous pouvez le personnaliser avec l'argument `cache_dir` sur
n'importe quelle méthode, où en spécifiant les variables d'environnement `HF_HOME` ou `HF_HUB_CACHE`.

Les modèles, datasets et espaces ont tous la même racine. Chacun de ces dépôts contient
le type de dépôt, le namespace (nom de l'organisation ou du nom d'utilisateur) s'il existe
et le nom du dépôt:

```
<CACHE_DIR>
├─ models--julien-c--EsperBERTo-small
├─ models--lysandrejik--arxiv-nlp
├─ models--bert-base-cased
├─ datasets--glue
├─ datasets--huggingface--DataMeasurementsFiles
├─ spaces--dalle-mini--dalle-mini
```

C'est parmi ces dossiers que tous les fichiers seront maintenant téléchargés depuis le Hub. Cacher
vous assure qu'un fichier n'est pas téléchargé deux fois s'il a déjà été téléchargé et qu'il n'a
pas été mis à jour; s'il a été mis à jour et que vous cherchez le dernier fichier, alors il téléchargera
le dernier fichier (tout en gardant les fichiers précédents intacts au cas où vous en auriez besoin).

Pour ce faire, tous les dossiers contiennent le même squelette:

```
<CACHE_DIR>
├─ datasets--glue
│  ├─ refs
│  ├─ blobs
│  ├─ snapshots
...
```

Chaque dossier est fait pour contenir les dossiers suivants:

### Refs

Le fichier `refs` contient des dossiers qui indiquent la dernière révision d'une référence donnée. Par
exemple, si précédemment, nous avions ajouté un fichier depuis la branche `main` d'un dépôt, le dossier
`refs` contiendra un fichier nommé `main`, qui lui même contiendra l'identifier de commit du head actuel.

Si le dernier commit de `main` a pour identifier `aaaaaa`, alors le fichier dans ``refs`
contiendra `aaaaaa`.

Si cette même branche est mise à jour avec un nouveau commit, qui a `bbbbbb` en tant
qu'identifier, alors re-télécharger un fichier de cette référence mettra à jour le fichier
`refs/main` afin qu'il contienne `bbbbbb`.

### Blobs

Le dossier `blobs` contient les fichiers que nous avons téléchargé. Le nom de chaque fichier est
son hash.

### Snapshots

Le dossier `snapshots` contient des symlinks vers les blobs mentionnés ci dessus. Il est lui même fait
de plusieurs dossiers:
un par révision connue!

Dans l'exemple ci-dessus, nous avons initialement ajouté un fichier depuis la révision `aaaaaa`, avant d'ajouter
un fichier basé sur la révision `bbbbbb`. Dans cette situation, nous aurions maintenant deux dossiers dans le
dossier `snapshots`: `aaaaaaa` et `bbbbbbb`.

Dans chacun de ces dossiers, il y a des symlinks qui ont le nom des fichiers que nous avons téléchargé. Par
exemple, si nous avions téléchargé le fichier `README.md` dans la révision `aaaaaa`, nous aurions ce chemin: 

```
<CACHE_DIR>/<REPO_NAME>/snapshots/aaaaaa/README.md
```

Ce fichier `README.md` est enfaite un symlink qui dirige vers le blob qui a le hash du fichier.

En créant le squelette de cette manière, nous ouvrons le mécanisme au partage de fichiers: si ce même
fichier était ajouté dans la révision `bbbbbb`, il aurait le même hash et le fichier n'aurait pas besoin
d'être re-téléchargé.

### .no_exist (avancé)

En plus des fichiers `blobs`, `refs` et `snapshots`, vous pourrez aussi trouver un dossier `.no_exist`
dans votre cache. Ce dossier garde une trace des fichiers que vous avez essayé de télécharger une fois
mais qui n'existent pas sur le Hub. Sa structure est la même que le dossier `snapshots` avec 1 sous-dossier
par révision connue:

```
<CACHE_DIR>/<REPO_NAME>/.no_exist/aaaaaa/config_inexistante.json
```

Contrairement au dossier `snapshots`, les fichiers sont de simples fichiers vides (sans symlinks).
Dans cet exemple, le fichier `"config_inexistante.json"` n'existe pas sur le Hub pour la révision
`"aaaaaa"`. Comme il ne sauvegarde que des fichiers vides, ce dossier est négligeable en terme d'utilisation
d'espace sur le disque.

Maintenant, vous vous demandez peut être, pourquoi cette information est elle pertinente ?
Dans certains cas, un framework essaye de charger des fichiers optionnels pour un modèle.
Enregistrer la non-existence d'un fichier optionnel rend le chargement d'un fichier plus
rapide vu qu'on économise 1 appel HTTP par fichier optionnel possible.
C'est par exemple le cas dans `transformers`, où chacun des tokenizer peut accepter des fichiers additionnels.
La première fois que vous chargez le tokenizer sur votre machine, il mettra en cache quels fichiers
optionnels existent (et lesquels n'existent pas) pour faire en sorte que le chargement soit plus rapide
lors des prochaines initialisations.

Pour tester si un fichier est en cache en local (sans faire aucune requête HTTP), vous pouvez utiliser
le helper [`try_to_load_from_cache`]. Il retournera soit le chemin du fichier (s'il existe est qu'il est
dans le cache), soit l'objet `_CACHED_NO_EXIST` (si la non existence est en cache), soit `None`
(si on ne sait pas).

```python
from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

filepath = try_to_load_from_cache()
if isinstance(filepath, str):
    # Le fichier existe et est dans le cache
    ...
elif filepath is _CACHED_NO_EXIST:
    # La non-existence du fichier est dans le cache
    ...
else:
    # Le fichier n'est pas dans le cache
    ...
```

### En pratique

En pratique, votre cache devrait ressembler à l'arbre suivant:

```text
    [  96]  .
    └── [ 160]  models--julien-c--EsperBERTo-small
        ├── [ 160]  blobs
        │   ├── [321M]  403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
        │   ├── [ 398]  7cb18dc9bafbfcf74629a4b760af1b160957a83e
        │   └── [1.4K]  d7edf6bd2a681fb0175f7735299831ee1b22b812
        ├── [  96]  refs
        │   └── [  40]  main
        └── [ 128]  snapshots
            ├── [ 128]  2439f60ef33a0d46d85da5001d52aeda5b00ce9f
            │   ├── [  52]  README.md -> ../../blobs/d7edf6bd2a681fb0175f7735299831ee1b22b812
            │   └── [  76]  pytorch_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
            └── [ 128]  bbc77c8132af1cc5cf678da3f1ddf2de43606d48
                ├── [  52]  README.md -> ../../blobs/7cb18dc9bafbfcf74629a4b760af1b160957a83e
                └── [  76]  pytorch_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
```

### Limitations

Afin d'avoir un système de cache efficace, `huggingface_hub` utilise les symlinks.
Cependant, les symlinks ne sont pas acceptés avec toutes les machines. C'est une
limitation connue en particulier sur Windows. Lorsque c'est le cas, `huggingface_hub`
n'utilise pas le chemin `blobs/` à la plce, elle enregistre les fichiers directement dans
`snapshots/`. Ceci permet aux utilisateurs de télécharger et mettre en cache des fichiers
directement depuis le Hub de la même manière que si tout marchait. Les outils pour
inspecter et supprimer le cache (voir ci-deccous) sont aussi fonctionnels. Toutefois,
le cache-system est moins efficace vu qu'un fichier risque d'être téléchargé un grand
nombre de fois si plusieurs révisions du même dépôt sont téléchargés.

Si vous voulez bénéficier d'un cache-system basé sur symlink sur une machine Windows,
vous avez le choix entre [activer le mode développeur](https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development)
ou lancer Python en tant qu'administrateur.

Lorsque les symlinks ne sont pas supportés, un message d'avertissement est affiché
à l'utilisateur afin de les prévenir qu'ils utilisent une version dégradée du
cache-system. Cet avertissement peut être désactivé en attribuant la valeur
"true" à la varialbe d'environnement `HF_HUB_DISABLE_SYMLINKS_WARNING`.

## Les assets

En plus de pouvoir mettre en cache des fichiers du Hub, les librairies demandent souvent
de mettre en cache d'autres fichiers liés à HF mais pas gérés directement par
`huggingface_hub` (par exemple: les fichiers téléchargés depuis GitHub, des données
pré-nettoyés, les logs,...). Afin de mettre en cache ces fichiers appelés `assets`,
[`cached_assets_path`] peut s'avérer utile. Ce petit helper génère des chemins dans le
cache HF d'une manière unifiée selon sur le nom de la librairie qui le demande et
peut aussi générer un chemin sur un namespace ou un nom de sous-dossier. Le but est de
permettre à toutes les librairies de gérer ses assets de sa propre manière
(i.e. pas de règle sur la structure) tant que c'est resté dans le bon dossier
d'assets. Ces librairies peuvent s'appuyer sur des outil d'`huggingface_hub` pour gérer
le cache, en partiluier pour scanner et supprimer des parties d'assets grace à une 
commande du CLI. 

```py
from huggingface_hub import cached_assets_path

assets_path = cached_assets_path(library_name="datasets", namespace="SQuAD", subfolder="download")
something_path = assets_path / "something.json" # Faites ce que vous voulez dans votre dossier d'assets !
```

<Tip>

[`cached_assets_path`] est la manière recommandé de sauvegarder des assets, mais vous
n'êtes pas obligés de l'utiliser. Si votre librairie utilise déjà son propre cache,
n'hésitez pas à l'utiliser!

</Tip>

### Les assets en pratique

En pratique, votre cache d'asset devrait ressembler à l'arbre suivant:

```text
    assets/
    └── datasets/
    │   ├── SQuAD/
    │   │   ├── downloaded/
    │   │   ├── extracted/
    │   │   └── processed/
    │   ├── Helsinki-NLP--tatoeba_mt/
    │       ├── downloaded/
    │       ├── extracted/
    │       └── processed/
    └── transformers/
        ├── default/
        │   ├── something/
        ├── bert-base-cased/
        │   ├── default/
        │   └── training/
    hub/
    └── models--julien-c--EsperBERTo-small/
        ├── blobs/
        │   ├── (...)
        │   ├── (...)
        ├── refs/
        │   └── (...)
        └── [ 128]  snapshots/
            ├── 2439f60ef33a0d46d85da5001d52aeda5b00ce9f/
            │   ├── (...)
            └── bbc77c8132af1cc5cf678da3f1ddf2de43606d48/
                └── (...)
```

## Scannez votre cache

Pour l'instant, les fichiers en cache ne sont jamais supprimés de votre chemin local:
lorsque vous téléchargez une nouvelle révision de la branche, les fichiers précédents
sont gardés au cas où vous en auriez encore besoin. Par conséquent, il peut être utile
de scanner votre chemin où se trouvent le cache afin de savoir quel dépôts et
révisions prennent le plus de place sur votre disque. `huggingface_hub` fournit
un helper pour effectuer ce scan qui peut être utilisé via `huggingface-cli`
où un script Python.


### Scannez le cache depuis le terminal

La manière la plus simple de scanner votre cache-system HF est d'utiliser la
commande `scan-cache` depuis l'outil `huggingface-clie`. CEtte commande scan le cache
et affiche un rapport avec des informations telles ques l'id du dépôt, le type de
dépôt, l'utilisation du disque, des références et un chemin local complet.

Le snippet ci-dessous montre le rapport d'un scan dans un dossier qui contient 4
modèles et 2 datasets en cache.

```text
➜ huggingface-cli scan-cache
REPO ID                     REPO TYPE SIZE ON DISK NB FILES LAST_ACCESSED LAST_MODIFIED REFS                LOCAL PATH
--------------------------- --------- ------------ -------- ------------- ------------- ------------------- -------------------------------------------------------------------------
glue                        dataset         116.3K       15 4 days ago    4 days ago    2.4.0, main, 1.17.0 /home/wauplin/.cache/huggingface/hub/datasets--glue
google/fleurs               dataset          64.9M        6 1 week ago    1 week ago    refs/pr/1, main     /home/wauplin/.cache/huggingface/hub/datasets--google--fleurs
Jean-Baptiste/camembert-ner model           441.0M        7 2 weeks ago   16 hours ago  main                /home/wauplin/.cache/huggingface/hub/models--Jean-Baptiste--camembert-ner
bert-base-cased             model             1.9G       13 1 week ago    2 years ago                       /home/wauplin/.cache/huggingface/hub/models--bert-base-cased
t5-base                     model            10.1K        3 3 months ago  3 months ago  main                /home/wauplin/.cache/huggingface/hub/models--t5-base
t5-small                    model           970.7M       11 3 days ago    3 days ago    refs/pr/1, main     /home/wauplin/.cache/huggingface/hub/models--t5-small

Done in 0.0s. Scanned 6 repo(s) for a total of 3.4G.
Got 1 warning(s) while scanning. Use -vvv to print details.
```

Pour avoir un rapport plus détaillé, utilisez l'option `--verbose`. Pour chacun des
dépôts, vous obtenez une liste de toutes les révisions qui ont été téléchargées. Comme
expliqué ci-dessus, les fichiers qui ne changent pas entre 2 révisions sont partagés
grâce aux symlinks. Ceci signifie que la taille du dépôt sur le disque doit être plus
petite que la somme des tailles de chacune de ses révisions. Par exemple, ici,
`bert-based-cased` a 2 révisions de 1.4G et 1.5G, mais l'utilisation totale du disque est
uniquement de 1.9G.

```text
➜ huggingface-cli scan-cache -v
REPO ID                     REPO TYPE REVISION                                 SIZE ON DISK NB FILES LAST_MODIFIED REFS        LOCAL PATH
--------------------------- --------- ---------------------------------------- ------------ -------- ------------- ----------- ----------------------------------------------------------------------------------------------------------------------------
glue                        dataset   9338f7b671827df886678df2bdd7cc7b4f36dffd        97.7K       14 4 days ago    main, 2.4.0 /home/wauplin/.cache/huggingface/hub/datasets--glue/snapshots/9338f7b671827df886678df2bdd7cc7b4f36dffd
glue                        dataset   f021ae41c879fcabcf823648ec685e3fead91fe7        97.8K       14 1 week ago    1.17.0      /home/wauplin/.cache/huggingface/hub/datasets--glue/snapshots/f021ae41c879fcabcf823648ec685e3fead91fe7
google/fleurs               dataset   129b6e96cf1967cd5d2b9b6aec75ce6cce7c89e8        25.4K        3 2 weeks ago   refs/pr/1   /home/wauplin/.cache/huggingface/hub/datasets--google--fleurs/snapshots/129b6e96cf1967cd5d2b9b6aec75ce6cce7c89e8
google/fleurs               dataset   24f85a01eb955224ca3946e70050869c56446805        64.9M        4 1 week ago    main        /home/wauplin/.cache/huggingface/hub/datasets--google--fleurs/snapshots/24f85a01eb955224ca3946e70050869c56446805
Jean-Baptiste/camembert-ner model     dbec8489a1c44ecad9da8a9185115bccabd799fe       441.0M        7 16 hours ago  main        /home/wauplin/.cache/huggingface/hub/models--Jean-Baptiste--camembert-ner/snapshots/dbec8489a1c44ecad9da8a9185115bccabd799fe
bert-base-cased             model     378aa1bda6387fd00e824948ebe3488630ad8565         1.5G        9 2 years ago               /home/wauplin/.cache/huggingface/hub/models--bert-base-cased/snapshots/378aa1bda6387fd00e824948ebe3488630ad8565
bert-base-cased             model     a8d257ba9925ef39f3036bfc338acf5283c512d9         1.4G        9 3 days ago    main        /home/wauplin/.cache/huggingface/hub/models--bert-base-cased/snapshots/a8d257ba9925ef39f3036bfc338acf5283c512d9
t5-base                     model     23aa4f41cb7c08d4b05c8f327b22bfa0eb8c7ad9        10.1K        3 1 week ago    main        /home/wauplin/.cache/huggingface/hub/models--t5-base/snapshots/23aa4f41cb7c08d4b05c8f327b22bfa0eb8c7ad9
t5-small                    model     98ffebbb27340ec1b1abd7c45da12c253ee1882a       726.2M        6 1 week ago    refs/pr/1   /home/wauplin/.cache/huggingface/hub/models--t5-small/snapshots/98ffebbb27340ec1b1abd7c45da12c253ee1882a
t5-small                    model     d0a119eedb3718e34c648e594394474cf95e0617       485.8M        6 4 weeks ago               /home/wauplin/.cache/huggingface/hub/models--t5-small/snapshots/d0a119eedb3718e34c648e594394474cf95e0617
t5-small                    model     d78aea13fa7ecd06c29e3e46195d6341255065d5       970.7M        9 1 week ago    main        /home/wauplin/.cache/huggingface/hub/models--t5-small/snapshots/d78aea13fa7ecd06c29e3e46195d6341255065d5

Done in 0.0s. Scanned 6 repo(s) for a total of 3.4G.
Got 1 warning(s) while scanning. Use -vvv to print details.
```

#### Exemple de grep

Vu que l'output de la commande est sous forme de donnée tabulaire, vous pouvez le combiner
avec n'importe quel outil similaire à `grep` pour filtrer les entrées. Voici un exemple
pour filtrer uniquement les révision du modèle "t5-small" sur une machine basée sur
Unix.

```text
➜ eval "huggingface-cli scan-cache -v" | grep "t5-small"
t5-small                    model     98ffebbb27340ec1b1abd7c45da12c253ee1882a       726.2M        6 1 week ago    refs/pr/1   /home/wauplin/.cache/huggingface/hub/models--t5-small/snapshots/98ffebbb27340ec1b1abd7c45da12c253ee1882a
t5-small                    model     d0a119eedb3718e34c648e594394474cf95e0617       485.8M        6 4 weeks ago               /home/wauplin/.cache/huggingface/hub/models--t5-small/snapshots/d0a119eedb3718e34c648e594394474cf95e0617
t5-small                    model     d78aea13fa7ecd06c29e3e46195d6341255065d5       970.7M        9 1 week ago    main        /home/wauplin/.cache/huggingface/hub/models--t5-small/snapshots/d78aea13fa7ecd06c29e3e46195d6341255065d5
```

### Scannez le cache depuis Python

Pour une utilisation plus avancée, utilisez [`scan_cache_dir`] qui est la fonction Python
appelée par l'outil du CLI

Vous pouvez l'utiliser pour avoir un rapport structuré autour des 4 dataclasses:

- [`HFCacheInfo`]: rapport complet retourné par [`scan_cache_dir`]
- [`CachedRepoInfo`]: informations sur le dépôt en cache
- [`CachedRevisionInfo`]: informations sur une révision en cache (i.e. "snapshot") à
  l'intérieur d'un dépôt
- [`CachedFileInfo`]: informations sur un fichier en cache dans une snapshot

Voici un exemple simple d'utilisation. Consultez les références pour plus de détails.

```py
>>> from huggingface_hub import scan_cache_dir

>>> hf_cache_info = scan_cache_dir()
HFCacheInfo(
    size_on_disk=3398085269,
    repos=frozenset({
        CachedRepoInfo(
            repo_id='t5-small',
            repo_type='model',
            repo_path=PosixPath(...),
            size_on_disk=970726914,
            nb_files=11,
            last_accessed=1662971707.3567169,
            last_modified=1662971107.3567169,
            revisions=frozenset({
                CachedRevisionInfo(
                    commit_hash='d78aea13fa7ecd06c29e3e46195d6341255065d5',
                    size_on_disk=970726339,
                    snapshot_path=PosixPath(...),
                    # Pas de `last_accessed` car les blobs sont partagés entre les révisions
                    last_modified=1662971107.3567169,
                    files=frozenset({
                        CachedFileInfo(
                            file_name='config.json',
                            size_on_disk=1197
                            file_path=PosixPath(...),
                            blob_path=PosixPath(...),
                            blob_last_accessed=1662971707.3567169,
                            blob_last_modified=1662971107.3567169,
                        ),
                        CachedFileInfo(...),
                        ...
                    }),
                ),
                CachedRevisionInfo(...),
                ...
            }),
        ),
        CachedRepoInfo(...),
        ...
    }),
    warnings=[
        CorruptedCacheException("Le chemin vers les snapshots n'existe par dans les dépôts en cache: ..."),
        CorruptedCacheException(...),
        ...
    ],
)
```

## Néttoyez votre cache

Scanner votre cache est intéressant mais après, vous aurez surement envie de supprimer
certaines parties du cache pour libérer de l'espace dans votre drive. C'est faisable
en utilisant la commande CLI `delete-cache`. L'helper [`~HFCacheInfo.delete_revisions`]
peut aussi être utilisé depuis le code depuis l'objet [`HFCacheInfo`] retourné lors
du scan du cache.

### Stratégie de suppression

Pour supprimer des dossiers du cache, vous devez passer une liste de révisions à
supprimer. L'outil définira une stratégie pour libérer de l'espace basé sur cette
liste. Il renvoie un objet [`DeleteCacheStrategy`] qui décrit les fichiers et dossiers
qui seront supprimés. [`DeleteCacheStrategy`] vous donne l'espace qui devrait être
libéré. Une fois que vous acceptez la suppression, vous devez l'exécuter pour que la
suppression soit effective. Afin d'éviter les différentces, vous ne pouvez pas modifier
manuellement un objet stratégie.

La stratégie pour supprimer des révisions est la suivante:

- Le dossier `snapshot` contenant les symlinks des révisions est supprimé.
- Les fichiers blobs qui sont visés uniquement par les révisions à supprimer sont supprimés aussi.
- Si une révision est lié à une `refs` ou plus, les références sont supprimées.
- Si toutes les révisions d'un dépôt sont supprimées, le dépôts en cache est supprimé.

<Tip>

Les hash de révision sont uniques parmi tous les dépôts. Ceci signifie que
vous n'avez pas besoin de fournir un `repo_id` ou un `repo_type` lors de la
suppression d'une révision.
</Tip>

<Tip warning={true}>

Si une révision n'est pas trouvée dans le cache, elle sera ignorée. En dehors de ça,
si un fichier où un dossier ne peut pas être trouvé lorsque vous essayez de le supprimer,
un avertissement sera affiché mais aucune erreur ne sera retournée. La suppression
continue pour d'autres chemins contenus dans l'objet [`DeleteCacheStrategy`].

</Tip>

### Nettoyez le cache depuis le terminal

La manière la plus simple de supprimer des révision de votre cache-system HF est
d'utiliser la commande `delete-cache` depuis l'outil `huggingface-cli`. Cette
commande a deux modes. Par défaut, un TUI (Terminla User Interface) es affiché
à l'utilisateur pour sélectionner la révision à supprimer. Ce TUI est actuellement
en beta car il n'a pas été testé sur toutes les plateformes. Si le TUI ne marche pas
sur votre machine, vous pouvez le désactiver en utilisant le flag `--disable-tui`.

#### Utilisation du TUI

C'est le mode par défaut. Pour l'utliser, vous avez d'abord besoin d'installer les
dépendances supplémentaire en lançant la commande suivante:

```
pip install huggingface_hub["cli"]
```

Ensuite lancez la commande:

```
huggingface-cli delete-cache
```

Vous devriez maintenant voir une liste de révisions que vous pouvez sélectionner/désélectionner:

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/delete-cache-tui.png"/>
</div>

Instructions:
    - Appuyez lsur les flèches `<haut>` et `<bas>` du clavier pour bouger le curseur.
    - Appuyez sur `<espace>` pour sélectionner/désélectionner un objet.
    - Lorsqu'une révision est sélectionnée, la première ligne est mise à jour pour vous montrer
      l'espace libéré
    - Appuyez sur `<entrée>` pour confirmer votre sélection.
    - Si vous voulez annuler l'opération et quitter, vous pouvez sélectionner le premier item
      ("none of the following"). Si cet item est sélectionné, le processus de suppression sera
      annulé, et ce, quel que soit les autres items sélectionnés. Sinon, vous pouvez aussi
      appuyer sur `<ctrl+c>` pour quitter le TUI.

Une fois que vous avez sélectionné les révision que vous voulez supprimer et que vous
avez appuyé sur `<entrée>`, un dernier message de confirmation sera affiché. Appuyez
sur `<entrée>` encore une fois et la suppression sera effective. Si vous voulez l'annuler,
appuyez sur `n`.

```txt
✗ huggingface-cli delete-cache --dir ~/.cache/huggingface/hub
? Select revisions to delete: 2 revision(s) selected.
? 2 revisions selected counting for 3.1G. Confirm deletion ? Yes
Start deletion.
Done. Deleted 1 repo(s) and 0 revision(s) for a total of 3.1G.
```

#### sans le TUI

Comme mentionné ci-dessus, le mode TUI est actuellement en beta et est optionnel. Il
se pourrait qu'il ne marche pas sur votre machine ou que vous ne le trouvez pas
pratique.

une autre approche est d'utiliser le flag `--disable-tui`. Le process est très similaire
a ce qu'on vous demandera pour review manuellement la liste des révisions à supprimer.
Cependant, cette étape manuelle ne se passera pas dans le terminal directement mais
dans un fichier temporaire généré sur le volet et que vous pourrez éditer manuellement.

Ce fichier a toutes les instructions dont vous avez besoin dans le header. Ouvrez le dans
votre éditeur de texte favoris. Pour sélectionner ou déselectionner une révision, commentez
ou décommentez simplement avec un `#`. Une fois que la review du manuel fini et que le fichier
est édité, vous pouvez le sauvegarder. Revenez à votre terminal et appuyez sur `<entrée>`.
Par défaut, l'espace libéré sera calculé avec la liste des révisions mise à jour. Vous
pouvez continuer de modifier le fichier ou confirmer avec `"y"`.

```sh
huggingface-cli delete-cache --disable-tui
```

Exemple de fichier de commande:
```txt
# INSTRUCTIONS
# ------------
# This is a temporary file created by running `huggingface-cli delete-cache` with the
# `--disable-tui` option. It contains a set of revisions that can be deleted from your
# local cache directory.
#
# Please manually review the revisions you want to delete:
#   - Revision hashes can be commented out with '#'.
#   - Only non-commented revisions in this file will be deleted.
#   - Revision hashes that are removed from this file are ignored as well.
#   - If `CANCEL_DELETION` line is uncommented, the all cache deletion is cancelled and
#     no changes will be applied.
#
# Once you've manually reviewed this file, please confirm deletion in the terminal. This
# file will be automatically removed once done.
# ------------

# KILL SWITCH
# ------------
# Un-comment following line to completely cancel the deletion process
# CANCEL_DELETION
# ------------

# REVISIONS
# ------------
# Dataset chrisjay/crowd-speech-africa (761.7M, used 5 days ago)
    ebedcd8c55c90d39fd27126d29d8484566cd27ca # Refs: main # modified 5 days ago

# Dataset oscar (3.3M, used 4 days ago)
#    916f956518279c5e60c63902ebdf3ddf9fa9d629 # Refs: main # modified 4 days ago

# Dataset wikiann (804.1K, used 2 weeks ago)
    89d089624b6323d69dcd9e5eb2def0551887a73a # Refs: main # modified 2 weeks ago

# Dataset z-uo/male-LJSpeech-italian (5.5G, used 5 days ago)
#    9cfa5647b32c0a30d0adfca06bf198d82192a0d1 # Refs: main # modified 5 days ago
```

### Nettoyez le cache depuis Python

Pour plus de flexibilité, vous pouvez aussi utiliser la méthode [`~HFCacheInfo.delete_revisions`]
depuis le code. Voici un exemple simple, consultez la référence pour plus de détails.

```py
>>> from huggingface_hub import scan_cache_dir

>>> delete_strategy = scan_cache_dir().delete_revisions(
...     "81fd1d6e7847c99f5862c9fb81387956d99ec7aa"
...     "e2983b237dccf3ab4937c97fa717319a9ca1a96d",
...     "6c0e6080953db56375760c0471a8c5f2929baf11",
... )
>>> print("Will free " + delete_strategy.expected_freed_size_str)
Will free 8.6G

>>> delete_strategy.execute()
Cache deletion done. Saved 8.6G.
```
