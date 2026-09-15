<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Comprendre la mise en cache

`huggingface_hub` utilise le disque local comme deux caches, qui évitent de retélécharger les éléments à nouveau. Le premier cache est un cache basé sur les fichiers, qui met en cache les fichiers individuels téléchargés depuis le Hub et garantit que le même fichier n'est pas téléchargé à nouveau lorsqu'un dépôt est mis à jour. Le deuxième cache est un cache de chunks, où chaque chunk représente une plage d'octets d'un fichier et garantit que les chunks partagés entre les fichiers ne sont téléchargés qu'une seule fois.

## Mise en cache basée sur les fichiers

Le système de cache du Hugging Face Hub est conçu pour être le cache central partagé entre les bibliothèques qui dépendent du Hub. Il a été mis à jour dans la v0.8.0 pour éviter de retélécharger les mêmes fichiers entre les révisions de version.

Le système de cache est conçu comme suit :

```
<CACHE_DIR>
├─ <MODELS>
├─ <DATASETS>
├─ <SPACES>
```

Le `<CACHE_DIR>` par défaut est `~/.cache/huggingface/hub`. Cependant, il est personnalisable avec l'argument `cache_dir` sur toutes les méthodes, ou en spécifiant soit la variable d'environnement `HF_HOME` soit `HF_HUB_CACHE`.

Les modèles, datasets et spaces partagent une racine commune. Chacun de ces dépôts contient le type de dépôt, le namespace (organisation ou nom d'utilisateur) s'il existe et le nom du dépôt :

```
<CACHE_DIR>
├─ models--julien-c--EsperBERTo-small
├─ models--lysandrejik--arxiv-nlp
├─ models--bert-base-cased
├─ datasets--glue
├─ datasets--huggingface--DataMeasurementsFiles
├─ spaces--dalle-mini--dalle-mini
```

C'est dans ces dossiers que tous les fichiers seront désormais téléchargés depuis le Hub. La mise en cache garantit qu'un fichier n'est pas téléchargé deux fois s'il existe déjà ou n'a pas été mis à jour ; mais s'il a été mis à jour, et que vous demandez le dernier fichier, alors il téléchargera le dernier fichier (tout en gardant le fichier précédent intact au cas où vous en auriez à nouveau besoin).

Pour y parvenir, tous les dossiers contiennent le même squelette :

```
<CACHE_DIR>
├─ datasets--glue
│  ├─ refs
│  ├─ blobs
│  ├─ snapshots
...
```

Chaque dossier est conçu pour contenir les éléments suivants :

### Refs

Le dossier `refs` contient des fichiers qui indiquent la dernière révision de la référence donnée. Par exemple, si nous avons précédemment récupéré un fichier depuis la branche `main` d'un dépôt, le dossier `refs` contiendra un fichier nommé `main`, qui contiendra lui-même l'identifiant de commit du head actuel.

Si le dernier commit de `main` a `aaaaaa` comme identifiant, alors il contiendra `aaaaaa`.

Si cette même branche est mise à jour avec un nouveau commit, qui a `bbbbbb` comme identifiant, alors retélécharger un fichier depuis cette référence mettra à jour le fichier `refs/main` pour contenir `bbbbbb`.

### Blobs

Le dossier `blobs` contient les fichiers réels que nous avons téléchargés. Le nom de chaque fichier est leur hash.

### Snapshots

Le dossier `snapshots` contient des liens symboliques vers les blobs mentionnés ci-dessus. Il est lui-même composé de plusieurs dossiers : un par révision connue !

Dans l'explication ci-dessus, nous avions initialement récupéré un fichier depuis la révision `aaaaaa`, avant de récupérer un fichier depuis la révision `bbbbbb`. Dans cette situation, nous aurions maintenant deux dossiers dans le dossier `snapshots` : `aaaaaa` et `bbbbbb`.

Dans chacun de ces dossiers, ont des liens symboliques sur les noms des fichiers que nous avons téléchargés. Par exemple, si nous avions téléchargé le fichier `README.md` à la révision `aaaaaa`, nous aurions le chemin suivant :

```
<CACHE_DIR>/<REPO_NAME>/snapshots/aaaaaa/README.md
```

Ce fichier `README.md` est en fait un lien symbolique pointant vers le blob qui a le hash du fichier.

En créant le squelette de cette façon, nous ouvrons le mécanisme au partage de fichiers : si le même fichier a été récupéré dans la révision `bbbbbb`, il aurait le même hash et le fichier n'aurait pas besoin d'être retéléchargé.

### .no_exist (avancé)

En plus des dossiers `blobs`, `refs` et `snapshots`, vous pourriez également trouver un dossier `.no_exist` dans votre cache. Ce dossier garde une trace des fichiers que vous avez essayé de télécharger une fois mais qui n'existent pas sur le Hub. Sa structure est la même que celle du dossier `snapshots` avec 1 sous-dossier par révision connue :

```
<CACHE_DIR>/<REPO_NAME>/.no_exist/aaaaaa/config_that_does_not_exist.json
```

Contrairement au dossier `snapshots`, les fichiers sont de simples fichiers vides (pas de liens symboliques). Dans cet exemple, le fichier `"config_that_does_not_exist.json"` n'existe pas sur le Hub pour la révision `"aaaaaa"`. Comme il ne stocke que des fichiers vides, ce dossier est négligeable en termes d'utilisation du disque.

Alors maintenant vous vous demandez peut-être, pourquoi cette information est-elle pertinente ? Dans certains cas, un framework essaie de charger des fichiers optionnels pour un modèle. Sauvegarder la non-existence de fichiers optionnels rend le chargement d'un modèle plus rapide car cela économise 1 appel HTTP par fichier optionnel possible. C'est par exemple le cas dans `transformers` où chaque tokenizer peut supporter des fichiers supplémentaires. La première fois que vous chargez le tokenizer sur votre machine, il mettra en cache quels fichiers optionnels existent (et lesquels n'existent pas) pour rendre le temps de chargement plus rapide pour les prochaines initialisations.

Pour tester si un fichier est mis en cache localement (sans faire aucune requête HTTP), vous pouvez utiliser le helper [`try_to_load_from_cache`]. Il retournera soit le chemin du fichier (s'il existe et est mis en cache), l'objet `_CACHED_NO_EXIST` (si la non-existence est mise en cache) ou `None` (si nous ne savons pas).

```python
from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

filepath = try_to_load_from_cache()
if isinstance(filepath, str):
    # le fichier existe et est mis en cache
    ...
elif filepath is _CACHED_NO_EXIST:
    # la non-existence du fichier est mise en cache
    ...
else:
    # le fichier n'est pas mis en cache
    ...
```

### En pratique

En pratique, votre cache devrait ressembler à l'arbre suivant :

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

Afin d'avoir un système de cache efficace, `huggingface-hub` utilise des liens symboliques. Cependant, les liens symboliques ne sont pas supportés sur toutes les machines. C'est une limitation connue en particulier sur Windows. Lorsque c'est le cas, `huggingface_hub` n'utilise pas le répertoire `blobs/` mais stocke directement les fichiers dans le répertoire `snapshots/` à la place. Cette solution de contournement permet aux utilisateurs de télécharger et mettre en cache des fichiers depuis le Hub exactement de la même manière. Les outils pour inspecter et supprimer le cache (voir ci-dessous) sont également supportés. Cependant, le système de cache est moins efficace car un seul fichier pourrait être téléchargé plusieurs fois si plusieurs révisions du même dépôt sont téléchargées.

Si vous souhaitez bénéficier du système de cache basé sur les liens symboliques sur une machine Windows, vous devez soit [activer le Mode Développeur](https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development) soit exécuter Python en tant qu'administrateur.

Lorsque les liens symboliques ne sont pas supportés, un message d'avertissement est affiché à l'utilisateur pour l'alerter qu'il utilise une version dégradée du système de cache. Cet avertissement peut être désactivé en définissant la variable d'environnement `HF_HUB_DISABLE_SYMLINKS_WARNING` à true.

## Mise en cache basée sur les chunks (Xet)

Pour fournir des transferts de fichiers plus efficaces, `hf_xet` ajoute un répertoire `xet` au cache `huggingface_hub` existant, créant une couche de cache supplémentaire pour permettre la déduplication basée sur les chunks. Ce cache contient des chunks (plages d'octets immuables de fichiers ~64KB de taille) et des shards (une structure de données qui mappe les fichiers aux chunks). Pour plus d'informations sur le système de stockage Xet, consultez cette [section](https://huggingface.co/docs/hub/xet/index).

Le répertoire `xet`, situé par défaut à `~/.cache/huggingface/xet`, contient deux caches, utilisés pour les uploads et les téléchargements. Il a la structure suivante :

```bash
<CACHE_DIR>
├─ xet
│  ├─ environment_identifier
│  │  ├─ chunk_cache
│  │  ├─ shard_cache
│  │  ├─ staging
```

Le répertoire `environment_identifier` est une chaîne encodée (il peut apparaître sur votre machine comme `https___cas_serv-tGqkUaZf_CBPHQ6h`). Ceci est utilisé pendant le développement permettant aux versions locales et de production du cache d'exister côte à côte simultanément. Il est également utilisé lors du téléchargement depuis des dépôts qui résident dans différentes [régions de stockage](https://huggingface.co/docs/hub/storage-regions). Vous pouvez voir plusieurs entrées de ce type dans le répertoire `xet`, chacune correspondant à un environnement différent, mais leur structure interne est la même.

Les répertoires internes servent les objectifs suivants :
* `chunk-cache` contient des chunks de données mis en cache qui sont utilisés pour accélérer les téléchargements.
* `shard-cache` contient des shards mis en cache qui sont utilisés sur le chemin d'upload.
* `staging` est un espace de travail conçu pour supporter les uploads reprenables.

Ceux-ci sont documentés ci-dessous.

Notez que le système de cache `xet`, comme le reste de `hf_xet`, est entièrement intégré avec `huggingface_hub`. Si vous utilisez les APIs existantes pour interagir avec les assets mis en cache, il n'y a pas besoin de mettre à jour votre workflow. Les caches `xet` sont construits comme une couche d'optimisation au-dessus de la déduplication basée sur les chunks de `hf_xet` existante et du système de cache `huggingface_hub`.

### `chunk_cache`

Ce cache est utilisé sur le chemin de téléchargement. La structure du répertoire de cache est basée sur un hash encodé en base-64 depuis le content-addressed store (CAS) qui soutient chaque dépôt activé par Xet. Un hash CAS sert de clé pour rechercher les offsets où les données sont stockées. Note : à partir de `hf_xet` 1.2.0, le chunk_cache est désactivé par défaut. Pour l'activer, définissez la variable d'environnement `HF_XET_CHUNK_CACHE_SIZE_BYTES` à la taille appropriée avant de lancer le processus Python.

Au niveau le plus élevé, les deux premières lettres du hash CAS encodé en base 64 sont utilisées pour créer un sous-répertoire dans le `chunk_cache` (les clés qui partagent ces deux premières lettres sont regroupées ici). Les niveaux internes sont composés de sous-répertoires avec la clé complète comme nom de répertoire. À la base se trouvent les éléments de cache qui sont des plages de blocs contenant les chunks mis en cache.

```bash
<CACHE_DIR>
├─ xet
│  ├─ chunk_cache
│  │  ├─ A1
│  │  │  ├─ A1GerURLUcISVivdseeoY1PnYifYkOaCCJ7V5Q9fjgxkZWZhdWx0
│  │  │  │  ├─ AAAAAAEAAAA5DQAAAAAAAIhRLjDI3SS5jYs4ysNKZiJy9XFI8CN7Ww0UyEA9KPD9
│  │  │  │  ├─ AQAAAAIAAABzngAAAAAAAPNqPjd5Zby5aBvabF7Z1itCx0ryMwoCnuQcDwq79jlB
```

Lors de la demande d'un fichier, la première chose que `hf_xet` fait est de communiquer avec le content addressed store (CAS) du stockage Xet pour les informations de reconstruction. Les informations de reconstruction contiennent des informations sur les clés CAS requises pour télécharger le fichier dans son intégralité.

Avant d'exécuter les requêtes pour les clés CAS, le `chunk_cache` est consulté. Si une clé dans le cache correspond à une clé CAS, alors il n'y a aucune raison d'émettre une requête pour ce contenu. `hf_xet` utilise les chunks stockés dans le répertoire à la place.

Comme le `chunk_cache` est purement une optimisation, pas une garantie, `hf_xet` utilise une politique d'éviction computationnellement efficace. Lorsque le `chunk_cache` est plein (voir `Limites et Limitations` ci-dessous), `hf_xet` implémente une politique d'éviction aléatoire lors de la sélection d'un candidat à l'éviction. Cela réduit significativement la surcharge de gestion d'un système de cache robuste (par exemple, LRU) tout en fournissant la plupart des avantages de la mise en cache des chunks.

### `shard_cache`

Ce cache est utilisé lors de l'upload de contenu vers le Hub. Le répertoire est plat, comprenant uniquement des fichiers shard, chacun utilisant un ID pour le nom du shard.

```sh
<CACHE_DIR>
├─ xet
│  ├─ shard_cache
│  │  ├─ 1fe4ffd5cf0c3375f1ef9aec5016cf773ccc5ca294293d3f92d92771dacfc15d.mdb
│  │  ├─ 906ee184dc1cd0615164a89ed64e8147b3fdccd1163d80d794c66814b3b09992.mdb
│  │  ├─ ceeeb7ea4cf6c0a8d395a2cf9c08871211fbbd17b9b5dc1005811845307e6b8f.mdb
│  │  ├─ e8535155b1b11ebd894c908e91a1e14e3461dddd1392695ddc90ae54a548d8b2.mdb
```

Le `shard_cache` contient des shards qui sont :

- Générés localement et uploadés avec succès vers le CAS
- Téléchargés depuis le CAS dans le cadre de l'algorithme de déduplication globale

Les shards fournissent un mappage entre les fichiers et les chunks. Pendant les uploads, chaque fichier est divisé en chunks et le hash du chunk est sauvegardé. Chaque shard dans le cache est ensuite consulté. Si un shard contient un hash de chunk qui est présent dans le fichier local en cours d'upload, alors ce chunk peut être supprimé car il est déjà stocké dans le CAS.

Tous les shards ont une date d'expiration de 3-4 semaines à partir du moment où ils sont téléchargés. Les shards qui sont expirés ne sont pas chargés pendant l'upload et sont supprimés une semaine après l'expiration.

### `staging`

Lorsqu'un upload se termine avant que le nouveau contenu ait été commité vers le dépôt, vous devrez reprendre le transfert de fichier. Cependant, il est possible que certains chunks aient été uploadés avec succès avant l'interruption.

Pour que vous n'ayez pas à recommencer depuis le début, le répertoire `staging` agit comme un espace de travail pendant les uploads, stockant les métadonnées pour les chunks uploadés avec succès. Le répertoire `staging` a la forme suivante :

```
<CACHE_DIR>
├─ xet
│  ├─ staging
│  │  ├─ shard-session
│  │  │  ├─ 906ee184dc1cd0615164a89ed64e8147b3fdccd1163d80d794c66814b3b09992.mdb
│  │  │  ├─ xorb-metadata
│  │  │  │  ├─ 1fe4ffd5cf0c3375f1ef9aec5016cf773ccc5ca294293d3f92d92771dacfc15d.mdb
```

Au fur et à mesure que les fichiers sont traités et les chunks uploadés avec succès, leurs métadonnées sont stockées dans `xorb-metadata` comme un shard. Lors de la reprise d'une session d'upload, chaque fichier est traité à nouveau et les shards dans ce répertoire sont consultés. Tout contenu qui a été uploadé avec succès est sauté, et tout nouveau contenu est uploadé (et ses métadonnées sauvegardées).

Pendant ce temps, `shard-session` stocke les informations de fichiers et de chunks pour les fichiers traités. À la fin réussie d'un upload, le contenu de ces shards est déplacé vers le `shard-cache` plus persistant.

### Limites et Limitations

Le `chunk_cache` est limité à 10GB de taille tandis que le `shard_cache` a une limite souple de 4GB. Par conception, les deux caches sont sans APIs de haut niveau, bien que leur taille soit configurable via les variables d'environnement `HF_XET_CHUNK_CACHE_SIZE_BYTES` et `HF_XET_SHARD_CACHE_SIZE_LIMIT`.

Ces caches sont utilisés principalement pour faciliter la reconstruction (téléchargement) ou l'upload d'un fichier. Pour interagir avec les assets eux-mêmes, il est recommandé d'utiliser les [APIs du système de cache `huggingface_hub`](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).

Si vous avez besoin de récupérer l'espace utilisé par l'un ou l'autre cache ou si vous devez déboguer tout problème potentiel lié au cache, supprimez simplement le cache `xet` entièrement en exécutant `rm -rf ~/<cache_dir>/xet` où `<cache_dir>` est l'emplacement de votre cache Hugging Face, typiquement `~/.cache/huggingface`.

Pour en savoir plus sur le stockage Xet, consultez cette [section](https://huggingface.co/docs/hub/xet/index).

## Mise en cache des assets

En plus de mettre en cache les fichiers du Hub, les bibliothèques en aval nécessitent souvent de mettre en cache d'autres fichiers liés à HF mais non gérés directement par `huggingface_hub` (exemple : fichier téléchargé depuis GitHub, données prétraitées, logs,...). Pour mettre en cache ces fichiers, appelés `assets`, on peut utiliser [`cached_assets_path`]. Ce petit helper génère des chemins dans le cache HF de manière unifiée basée sur le nom de la bibliothèque qui le demande et optionnellement sur un namespace et un nom de sous-dossier. L'objectif est de laisser chaque bibliothèque en aval gérer ses assets à sa propre manière (par exemple pas de règle sur la structure) tant qu'elle reste dans le bon dossier d'assets. Ces bibliothèques peuvent alors exploiter les outils de `huggingface_hub` pour gérer le cache, en particulier scanner et supprimer des parties des assets depuis une commande CLI.

```py
from huggingface_hub import cached_assets_path

assets_path = cached_assets_path(library_name="datasets", namespace="SQuAD", subfolder="download")
something_path = assets_path / "something.json" # Faites ce que vous voulez dans votre dossier d'assets !
```

> [!TIP]
> [`cached_assets_path`] est la méthode recommandée pour stocker les assets mais n'est pas obligatoire. Si votre bibliothèque utilise déjà son propre cache, n'hésitez pas à l'utiliser !

### Assets en pratique

En pratique, votre cache d'assets devrait ressembler à l'arbre suivant :

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

## Gérer votre cache basé sur les fichiers

### Inspecter votre cache

Pour le moment, les fichiers mis en cache ne sont jamais supprimés de votre répertoire local : lorsque vous téléchargez une nouvelle révision d'une branche, les fichiers précédents sont conservés au cas où vous en auriez à nouveau besoin. Par conséquent, il peut être utile d'inspecter votre répertoire de cache afin de savoir quels dépôts et révisions occupent le plus d'espace disque. `huggingface_hub` fournit des helpers que vous pouvez utiliser depuis le CLI `hf` ou depuis Python.

**Inspecter le cache depuis le terminal**

Exécutez `hf cache ls` pour explorer ce qui est stocké localement. Par défaut, la commande agrège les informations par dépôt :

```text
➜ hf cache ls
ID                                   SIZE   LAST_ACCESSED LAST_MODIFIED REFS
------------------------------------ ------- ------------- ------------- -------------------
dataset/glue                         116.3K 4 days ago     4 days ago     2.4.0 main 1.17.0
dataset/google/fleurs                 64.9M 1 week ago     1 week ago     main refs/pr/1
model/Jean-Baptiste/camembert-ner    441.0M 2 weeks ago    16 hours ago   main
model/bert-base-cased                  1.9G 1 week ago     2 years ago
model/t5-base                          10.1K 3 months ago   3 months ago   main
model/t5-small                        970.7M 3 days ago     3 days ago     main refs/pr/1

Found 6 repo(s) for a total of 12 revision(s) and 3.4G on disk.
```

Ajoutez `--revisions` pour lister chaque snapshot mis en cache et enchaînez les filtres pour vous concentrer sur ce qui compte. Les filtres comprennent les tailles et durées lisibles par l'homme, donc des expressions telles que `size>1GB` ou `accessed>30d` fonctionnent directement :

```text
➜ hf cache ls --revisions --filter "size>1GB" --filter "accessed>30d"
ID                                   REVISION            SIZE   LAST_MODIFIED REFS
------------------------------------ ------------------ ------- ------------- -------------------
model/bert-base-cased                6d1d7a1a2a6cf4c2    1.9G  2 years ago
model/t5-small                       1c610f6b3f5e7d8a    1.1G  3 months ago  main

Found 2 repo(s) for a total of 2 revision(s) and 3.0G on disk.
```

Besoin d'une sortie lisible par machine ? Utilisez `--format json` pour obtenir des objets structurés ou `--format csv` pour des feuilles de calcul. Alternativement `--quiet` n'imprime que les identifiants (un par ligne) pour que vous puissiez les passer à d'autres outils. Utilisez `--sort` pour ordonner les entrées par `accessed`, `modified`, `name`, ou `size` (ajoutez `:asc` ou `:desc` pour contrôler l'ordre), et `--limit` pour restreindre les résultats aux N premières entrées. Combinez ces options avec `--cache-dir` lorsque vous devez inspecter un cache stocké en dehors de `HF_HOME`.

**Filtrer avec les outils shell courants**

La sortie tabulaire signifie que vous pouvez continuer à utiliser les outils que vous connaissez déjà. Par exemple, l'extrait ci-dessous trouve chaque révision mise en cache liée à `t5-small` :

```text
➜ eval "hf cache ls --revisions" | grep "t5-small"
model/t5-small                       1c610f6b3f5e7d8a    1.1G  3 months ago  main
model/t5-small                       8f3ad1c90fed7a62    820.1M 2 weeks ago   refs/pr/1
```

**Inspecter le cache depuis Python**

Pour une utilisation plus avancée, utilisez [`scan_cache_dir`] qui est l'utilitaire python appelé par l'outil CLI.

Vous pouvez l'utiliser pour obtenir un rapport détaillé structuré autour de 4 dataclasses :

- [`HFCacheInfo`] : rapport complet retourné par [`scan_cache_dir`]
- [`CachedRepoInfo`] : informations sur un dépôt mis en cache
- [`CachedRevisionInfo`] : informations sur une révision mise en cache (par exemple "snapshot") dans un dépôt
- [`CachedFileInfo`] : informations sur un fichier mis en cache dans un snapshot

Voici un exemple d'utilisation simple. Consultez la référence pour les détails.

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
        CorruptedCacheException("Snapshots dir doesn't exist in cached repo: ..."),
        CorruptedCacheException(...),
        ...
    ],
)
```

### Vérifier votre cache

`huggingface_hub` peut vérifier que vos fichiers mis en cache correspondent aux checksums sur le Hub. Utilisez le CLI `hf cache verify` pour valider la cohérence des fichiers pour une révision spécifique d'un dépôt spécifique :

```bash
>>> hf cache verify meta-llama/Llama-3.2-1B-Instruct
✅ Verified 13 file(s) for 'meta-llama/Llama-3.2-1B-Instruct' (model) in ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
  All checksums match.
```

Vérifiez une révision mise en cache spécifique :

```bash
>>> hf cache verify meta-llama/Llama-3.1-8B-Instruct --revision 0e9e39f249a16976918f6564b8830bc894c89659
```

> [!TIP]
> Consultez la [référence CLI `hf cache verify`](../package_reference/cli#hf-cache-verify) pour plus de détails sur l'utilisation et une liste complète des options.

### Nettoyer votre cache

Scanner votre cache est intéressant mais ce que vous voulez vraiment faire ensuite est généralement de supprimer certaines portions pour libérer de l'espace sur votre disque. Cela est possible en utilisant les commandes CLI `hf cache rm` et `hf cache prune`. On peut également utiliser avec un programme le helper [`~HFCacheInfo.delete_revisions`] de l'objet [`HFCacheInfo`] retourné lors du scan du cache.

**Stratégie de suppression**

Pour supprimer une partie du cache, vous devez passer une liste de révisions à supprimer. L'outil définira une stratégie pour libérer l'espace basée sur cette liste. Il retourne un objet [`DeleteCacheStrategy`] qui décrit quels fichiers et dossiers seront supprimés. Le [`DeleteCacheStrategy`] vous permet de savoir combien d'espace est prévu d'être libéré. Une fois que vous êtes d'accord avec la suppression, vous devez l'exécuter pour rendre la suppression effective. Afin d'éviter les divergences, vous ne pouvez pas éditer un objet stratégie manuellement.

La stratégie pour supprimer des révisions est la suivante :

- le dossier `snapshot` contenant les liens symboliques de révision est supprimé.
- les fichiers blobs qui sont ciblés uniquement par les révisions à supprimer sont également supprimés.
- si une révision est liée à 1 ou plusieurs `refs`, les références sont supprimées.
- si toutes les révisions d'un dépôt sont supprimées, le dépôt mis en cache entier est supprimé.

> [!TIP]
> Les hashs de révision sont uniques dans tous les dépôts. `hf cache rm` accepte donc soit un identifiant de dépôt (par exemple `model/bert-base-uncased`) soit un hash de révision seul ; lors du passage d'un hash, vous n'avez pas besoin de spécifier le dépôt séparément.

> [!WARNING]
> Si une révision n'est pas trouvée dans le cache, elle sera ignorée silencieusement. De plus, si un fichier ou dossier ne peut pas être trouvé lors de la tentative de suppression, un avertissement sera enregistré mais aucune erreur ne sera levée. La suppression continue pour les autres chemins contenus dans l'objet [`DeleteCacheStrategy`].

**Nettoyer le cache depuis le terminal**

Utilisez `hf cache rm` pour supprimer définitivement des dépôts ou révisions de votre cache. Passez un ou plusieurs identifiants de dépôt (par exemple `model/bert-base-uncased`) ou hashs de révision :

```text
➜ hf cache rm model/bert-base-cased
About to delete 1 repo(s) totalling 1.9G.
  - model/bert-base-cased (entire repo)
Proceed with deletion? [y/N]: y
Deleted 1 repo(s) and 1 revision(s); freed 1.9G.
```

Vous pouvez également utiliser `hf cache rm` en combinaison avec `hf cache ls --quiet` pour supprimer en masse les entrées identifiées par un filtre :

```bash
>>> hf cache rm $(hf cache ls --filter "accessed>1y" -q) -y
About to delete 2 repo(s) totalling 5.31G.
  - model/meta-llama/Llama-3.2-1B-Instruct (entire repo)
  - model/hexgrad/Kokoro-82M (entire repo)
Delete repo: ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct
Delete repo: ~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M
Cache deletion done. Saved 5.31G.
Deleted 2 repo(s) and 2 revision(s); freed 5.31G.
```

Mélangez des dépôts et révisions dans le même appel. Ajoutez `--dry-run` pour prévisualiser l'impact, ou `--yes` pour sauter l'invite de confirmation lors de l'écriture de scripts :

```text
➜ hf cache rm model/t5-small 8f3ad1c --dry-run
About to delete 1 repo(s) and 1 revision(s) totalling 1.1G.
  - model/t5-small:
      8f3ad1c [main] 1.1G
Dry run: no files were deleted.
```

Lorsque vous travaillez en dehors de l'emplacement de cache par défaut, associez la commande avec `--cache-dir PATH`.

Pour nettoyer les snapshots détachés en masse, exécutez `hf cache prune`. Il sélectionne automatiquement les révisions qui ne sont plus référencées par une branche ou un tag :

```text
➜ hf cache prune
About to delete 3 unreferenced revision(s) (2.4G total).
  - model/t5-small:
      1c610f6b [refs/pr/1] 820.1M
      d4ec9b72 [(detached)] 640.5M
  - dataset/google/fleurs:
      2b91c8dd [(detached)] 937.6M
Proceed? [y/N]: y
Deleted 3 unreferenced revision(s); freed 2.4G.
```

Les deux commandes supportent `--dry-run`, `--yes`, et `--cache-dir` afin que vous puissiez prévisualiser, automatiser et cibler des répertoires de cache alternatifs selon les besoins.

**Nettoyer le cache depuis Python**

Pour plus de flexibilité, vous pouvez également utiliser la méthode [`~HFCacheInfo.delete_revisions`] par programme. Voici un exemple simple. Consultez la référence pour les détails.

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
