<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Dateien aus dem Hub herunterladen

Die `huggingface_hub`-Bibliothek bietet Funktionen zum Herunterladen von Dateien aus den auf dem Hub gespeicherten Repositories. Sie können diese Funktionen unabhängig verwenden oder in Ihre eigene Bibliothek integrieren, um es Ihren Benutzern zu erleichtern, mit dem Hub zu interagieren. In diesem Leitfaden erfahren Sie, wie Sie:

* Einzelne Dateien herunterladen und zwischenspeichern.
* Ein gesamtes Repository herunterladen und zwischenspeichern.
* Dateien in einen lokalen Ordner herunterladen.

## Einzelne Dateien herunterladen

Die [`hf_hub_download`]-Funktion ist die Hauptfunktion zum Herunterladen von Dateien aus dem Hub. Sie lädt die Remote-Datei herunter, speichert sie auf der Festplatte (auf eine versionsbewusste Art und Weise) und gibt ihren lokalen Dateipfad zurück.

> [!TIP]
> Der zurückgegebene Dateipfad verweist auf den lokalen Cache von HF. Es ist daher wichtig, die Datei nicht zu ändern, um einen beschädigten Cache zu vermeiden. Wenn Sie mehr darüber erfahren möchten, wie Dateien zwischengespeichert werden, lesen Sie bitte unseren [Caching-Leitfaden](./manage-cache).

### Von der neuesten Version

Wählen Sie die Datei zum Herunterladen anhand der Parameter `repo_id`, `repo_type` und `filename` aus. Standardmäßig wird davon ausgegangen, dass die Datei Teil einer `model`-Repository ist.

```python
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json")
'/root/.cache/huggingface/hub/models--lysandre--arxiv-nlp/snapshots/894a9adde21d9a3e3843e6d5aeaaf01875c7fade/config.json'

# Herunterladen von einem Dataset
>>> hf_hub_download(repo_id="google/fleurs", filename="fleurs.py", repo_type="dataset")
'/root/.cache/huggingface/hub/datasets--google--fleurs/snapshots/199e4ae37915137c555b1765c01477c216287d34/fleurs.py'
```

### Von einer spezifischen Version

Standardmäßig wird die neueste Version vom Hauptzweig `main` heruntergeladen. In einigen Fällen möchten Sie jedoch eine Datei in einer bestimmten Version herunterladen (z. B. aus einem bestimmten Zweig, einem PR, einem Tag oder einem Commit-Hash). Verwenden Sie dazu den Parameter `revision`:

```python
# Herunterladen vom Tag `v1.0`
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="v1.0")

# Herunterladen vom Zweig `test-branch`
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="test-branch")

# Herunterladen von Pull Request #3
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="refs/pr/3")

# Herunterladen von einem spezifischen Commit-Hash
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="877b84a8f93f2d619faa2a6e514a32beef88ab0a")
```

**Hinweis:** Bei Verwendung des Commit-Hashs muss der vollständige Hash anstelle eines 7-Zeichen-Commit-Hashs verwendet werden.

### URL zum Herunterladen erstellen

Falls Sie die URL erstellen möchten, die zum Herunterladen einer Datei aus einem Repo verwendet wird, können Sie [`hf_hub_url`] verwenden, das eine URL zurückgibt. Beachten Sie, dass es intern von [`hf_hub_download`] verwendet wird.

## Gesamte Repository herunterladen

[`snapshot_download`] lädt ein gesamtes Repository zu einer bestimmten Revision herunter. Es verwendet intern [`hf_hub_download`], was bedeutet, dass alle heruntergeladenen Dateien auch auf Ihrer lokalen Festplatte zwischengespeichert werden. Die Downloads werden gleichzeitig durchgeführt, um den Prozess zu beschleunigen.

Um ein ganzes Repository herunterzuladen, geben Sie einfach die `repo_id` und `repo_type` an:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp")
'/home/lysandre/.cache/huggingface/hub/models--lysandre--arxiv-nlp/snapshots/894a9adde21d9a3e3843e6d5aeaaf01875c7fade'

# Oder von einem Dataset
>>> snapshot_download(repo_id="google/fleurs", repo_type="dataset")
'/home/lysandre/.cache/huggingface/hub/datasets--google--fleurs/snapshots/199e4ae37915137c555b1765c01477c216287d34'
```

[`snapshot_download`] lädt standardmäßig die neueste Revision herunter. Wenn Sie eine spezifische Repository-Revision wünschen, verwenden Sie den Parameter `revision`:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", revision="refs/pr/1")
```

### Dateien filtern zum Herunterladen

[`snapshot_download`] bietet eine einfache Möglichkeit, ein Repository herunterzuladen. Sie möchten jedoch nicht immer den gesamten Inhalt eines Repositories herunterladen. Beispielsweise möchten Sie möglicherweise verhindern, dass alle `.bin`-Dateien heruntergeladen werden, wenn Sie wissen, dass Sie nur die `.safetensors`-Gewichtungen verwenden werden. Dies können Sie mit den Parametern `allow_patterns` und `ignore_patterns` tun.

Diese Parameter akzeptieren entweder ein einzelnes Muster oder eine Liste von Mustern. Muster sind Standard-Wildcards (globbing patterns) wie [hier](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm) dokumentiert. Die Mustervergleichung basiert auf [`fnmatch`](https://docs.python.org/3/library/fnmatch.html).

Beispielsweise können Sie `allow_patterns` verwenden, um nur JSON-Konfigurationsdateien herunterzuladen:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", allow_patterns="*.json")
```

Andererseits können Sie mit `ignore_patterns` bestimmte Dateien vom Herunterladen ausschließen. Im folgenden Beispiel werden die Dateierweiterungen `.msgpack` und `.h5` ignoriert:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", ignore_patterns=["*.msgpack", "*.h5"])
```

Schließlich können Sie beide kombinieren, um Ihren Download genau zu filtern. Hier ist ein Beispiel, wie man alle json- und markdown-Dateien herunterlädt, außer `vocab.json`.

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="gpt2", allow_patterns=["*.md", "*.json"], ignore_patterns="vocab.json")
```

## Datei(en) in lokalen Ordner herunterladen

Die empfohlene (und standardmäßige) Methode zum Herunterladen von Dateien aus dem Hub besteht darin, das [Cache-System](./manage-cache) zu verwenden. Sie können Ihren Cache-Ort festlegen, indem Sie den `cache_dir`-Parameter setzen (sowohl in [`hf_hub_download`] als auch in [`snapshot_download`]).

In einigen Fällen möchten Sie jedoch Dateien herunterladen und in einen bestimmten Ordner verschieben. Dies ist nützlich, um einen Workflow zu erhalten, der den `git`-Befehlen ähnelt. Sie können dies mit den Parametern `local_dir` und `local_dir_use_symlinks` tun:

- local_dir muss ein Pfad zu einem Ordner auf Ihrem System sein. Die heruntergeladenen Dateien behalten dieselbe Dateistruktur wie im Repository. Wenn zum Beispiel `filename="data/train.csv"` und `local_dir="pfad/zum/ordner"` ist, wird der zurückgegebene Dateipfad `"pfad/zum/ordner/data/train.csv"` sein.
- `local_dir_use_symlinks` definiert, wie die Datei in Ihrem lokalen Ordner gespeichert werden muss.
  - Das Standardverhalten (`"auto"`) besteht darin, kleine Dateien (<5MB) zu duplizieren und für größere Dateien Symlinks zu verwenden. Symlinks ermöglichen
    die Optimierung von Bandbreite und Speicherplatz. Das manuelle Bearbeiten einer verlinkten Datei könnte jedoch den Cache beschädigen, daher die Duplizierung für kleine Dateien. Die 5-MB-Schwelle kann mit der `HF_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD`-Umgebungsvariable konfiguriert werden.
  - Wenn `local_dir_use_symlinks=True` gesetzt ist, werden alle Dateien verlinkt, um den Speicherplatz optimal zu nutzen. Dies ist zum Beispiel nützlich, wenn ein riesiges Dataset mit Tausenden von kleinen Dateien heruntergeladen wird.
  - Wenn Sie überhaupt keine Symlinks möchten, können Sie sie deaktivieren (`local_dir_use_symlinks=False`). Das Cache-Verzeichnis wird weiterhin verwendet, um zu überprüfen, ob die Datei bereits im Cache ist oder nicht. Wenn sie bereits im Cache ist, wird die Datei aus dem Cache **dupliziert** (d.h. Bandbreite wird gespart, aber der Speicherplatzverbrauch steigt). Wenn die Datei noch nicht im Cache ist, wird sie heruntergeladen und direkt in das lokale Verzeichnis verschoben. Das bedeutet, dass wenn Sie sie später woanders wiederverwenden müssen, sie **erneut heruntergeladen** wird.

Hier ist eine Tabelle, die die verschiedenen Optionen zusammenfasst, um Ihnen zu helfen, die Parameter zu wählen, die am besten zu Ihrem Anwendungsfall passen.

<!-- Generated with https://www.tablesgenerator.com/markdown_tables -->
| Parameter | Datei schon im Cache | Zurückgegebener Pfad | Pfad lesbar? | Kann im Pfad speichern? | Optimierter Datendurchsatz | Optimierter Speicherplatz |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `local_dir=None` |  | Symlink im Cache | ✅ | ❌<br>_(Speichern würde den Cache beschädigen)_ | ✅ | ✅ |
| `local_dir="path/to/folder"`<br>`local_dir_use_symlinks="auto"` |  | Datei oder Symlink im Ordner | ✅ | ✅ _(für kleine Dateien)_ <br> ⚠️ _(für große Dateien den Pfad nicht auflösen vor dem Speichern)_ | ✅ | ✅ |
| `local_dir="path/to/folder"`<br>`local_dir_use_symlinks=True` |  | Symlink im Ordner | ✅ | ⚠️<br>_(den Pfad nicht auflösen vor dem Speichern)_ | ✅ | ✅ |
| `local_dir="path/to/folder"`<br>`local_dir_use_symlinks=False` | Nein | Datei im Ordner | ✅ | ✅ | ❌<br>_(bei erneutem Ausführen wird die Datei erneut heruntergeladen)_ | ⚠️<br>(mehrere Kopien, wenn in mehreren Ordnern ausgeführt) |
| `local_dir="path/to/folder"`<br>`local_dir_use_symlinks=False` | Ja | Datei im Ordner | ✅ | ✅ | ⚠️<br>_(Datei muss zuerst im Cache gespeichert werden)_ | ❌<br>_(Datei wird dupliziert)_ |

**Hinweis**: Wenn Sie einen Windows-Computer verwenden, müssen Sie den Entwicklermodus aktivieren oder `huggingface_hub` als Administrator ausführen, um Symlinks zu aktivieren. Weitere Details finden Sie im Abschnitt über [Cache-Beschränkungen](../guides/manage-cache#limitations).

## Herunterladen mit dem CLI

Sie können den `hf download`-Befehl im Terminal verwenden, um Dateien direkt aus dem Hub herunterzuladen. Intern verwendet es die gleichen [`hf_hub_download`] und [`snapshot_download`] Helfer, die oben beschrieben wurden, und gibt den zurückgegebenen Pfad im Terminal aus:

```bash
>>> hf download gpt2 config.json
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

Standardmäßig wird das lokal gespeicherte Token (mit `hf auth login`) verwendet. Wenn Sie sich ausdrücklich authentifizieren möchten, verwenden Sie die `--token` Option:

```bash
>>> hf download gpt2 config.json --token=hf_****
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

Sie können mehrere Dateien gleichzeitig herunterladen, wobei eine Fortschrittsleiste angezeigt wird und der Snapshot-Pfad zurückgegeben wird, in dem sich die Dateien befinden:

```bash
>>> hf download gpt2 config.json model.safetensors
Fetching 2 files: 100%|████████████████████████████████████████████| 2/2 [00:00<00:00, 23831.27it/s]
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

Wenn Sie die Fortschrittsleisten und mögliche Warnungen stummschalten möchten, verwenden Sie die Option `--quiet`. Dies kann nützlich sein, wenn Sie die Ausgabe an einen anderen Befehl in einem Skript weitergeben möchten.

```bash
>>> hf download gpt2 config.json model.safetensors
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

Standardmäßig werden Dateien im Cache-Verzeichnis heruntergeladen, das durch die Umgebungsvariable `HF_HOME` definiert ist (oder `~/.cache/huggingface/hub`, wenn nicht angegeben). Sie können dies mit der Option `--cache-dir` überschreiben:

```bash
>>> hf download gpt2 config.json --cache-dir=./cache
./cache/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

Wenn Sie Dateien in einen lokalen Ordner herunterladen möchten, ohne die Cache-Verzeichnisstruktur, können Sie `--local-dir` verwenden. Das Herunterladen in einen lokalen Ordner hat seine Einschränkungen, die in dieser [Tabelle](https://huggingface.co/docs/huggingface_hub/guides/download#download-files-to-local-folder) aufgeführt sind.

```bash
>>> hf download gpt2 config.json --local-dir=./models/gpt2
./models/gpt2/config.json
```

Es gibt weitere Argumente, die Sie angeben können, um aus verschiedenen Repo-Typen oder Revisionen herunterzuladen und Dateien zum Herunterladen mit Glob-Mustern ein- oder auszuschließen:

```bash
>>> hf download bigcode/the-stack --repo-type=dataset --revision=v1.2 --include="data/python/*" --exclu
de="*.json" --exclude="*.zip"
Fetching 206 files:   100%|████████████████████████████████████████████| 206/206 [02:31<2:31, ?it/s]
/home/wauplin/.cache/huggingface/hub/datasets--bigcode--the-stack/snapshots/9ca8fa6acdbc8ce920a0cb58adcdafc495818ae7
```

Für eine vollständige Liste der Argumente führen Sie bitte den folgenden Befehl aus:

```bash
hf download --help
```
