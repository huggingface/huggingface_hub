<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Verwalten des `huggingface_hub` Cache-Systems

## Caching verstehen

Das Hugging Face Hub Cache-System wurde entwickelt, um der zentrale Cache zu sein,
der zwischen Bibliotheken geteilt wird, welche vom Hub abhängen. Es wurde in v0.8.0 aktualisiert,
um das erneute Herunterladen von Dateien zwischen Revisionen zu verhindern.

Das Cache-System ist wie folgt aufgebaut:

```
<CACHE_DIR>
├─ <MODELS>
├─ <DATASETS>
├─ <SPACES>
```

Der `<CACHE_DIR>` ist normalerweise das Home-Verzeichnis Ihres Benutzers. Es kann jedoch mit dem
`cache_dir`-Argument in allen Methoden oder durch Angabe der Umgebungsvariablen
`HF_HOME` oder `HF_HUB_CACHE` angepasst werden.

Modelle, Datensätze und Räume teilen eine gemeinsame Wurzel.
Jedes dieser Repositories enthält den Repository-Typ, den Namensraum (Organisation oder Benutzername),
falls vorhanden, und den Repository-Namen:

```
<CACHE_DIR>
├─ models--julien-c--EsperBERTo-small
├─ models--lysandrejik--arxiv-nlp
├─ models--bert-base-cased
├─ datasets--glue
├─ datasets--huggingface--DataMeasurementsFiles
├─ spaces--dalle-mini--dalle-mini
```

Innerhalb dieser Ordner werden nun alle Dateien vom Hub heruntergeladen. Das Caching stellt sicher,
dass eine Datei nicht zweimal heruntergeladen wird, wenn sie bereits existiert und nicht aktualisiert wurde;
wurde sie jedoch aktualisiert und Sie fordern die neueste Datei an, wird die neueste Datei heruntergeladen
(während die vorherige Datei intakt bleibt, falls Sie sie erneut benötigen).

Um dies zu erreichen, enthalten alle Ordner dasselbe Grundgerüst:

```
<CACHE_DIR>
├─ datasets--glue
│  ├─ refs
│  ├─ blobs
│  ├─ snapshots
...
```

Jeder Ordner ist so gestaltet, dass er das Folgende enthält:

### Refs

Der Ordner `refs` enthält Dateien, die die neueste Revision des gegebenen Verweises anzeigen.
Zum Beispiel, wenn wir zuvor eine Datei aus dem `main`-Branch eines Repositories abgerufen haben,
wird der Ordner `refs` eine Datei namens `main` enthalten, die selbst den Commit-Identifikator der aktuellen HEAD-Branch enthält.

Wenn der neueste Commit von `main` den Identifikator `aaaaaa` hat, dann enthält er `aaaaaa`.

Wenn derselbe Zweig mit einem neuen Commit aktualisiert wird, der den Identifikator `bbbbbb` hat,
wird das erneute Herunterladen einer Datei von diesem Verweis die Datei `refs/main` aktualisieren, um `bbbbbb` zu enthalten.

### Blobs

Der Ordner `blobs` enthält die tatsächlichen Dateien, die wir heruntergeladen haben. Der Name jeder Datei ist ihr Hash.

### Snapshots

Der Ordner `snapshots` enthält Symlinks zu den oben erwähnten Blobs.
Er besteht selbst aus mehreren Ordnern: einem pro bekannter Revision!

In der obigen Erklärung hatten wir zunächst eine Datei von der Revision `aaaaaa` abgerufen, bevor wir eine Datei
von der Revision `bbbbbb` abgerufen haben. In dieser Situation hätten wir jetzt zwei Ordner im Ordner `snapshots`: `aaaaaa` und `bbbbbb`.

In jedem dieser Ordner leben Symlinks, die die Namen der Dateien haben, die wir heruntergeladen haben.
Wenn wir zum Beispiel die Datei `README.md` in der Revision `aaaaaa` heruntergeladen hätten, hätten wir den folgenden Pfad:

```
<CACHE_DIR>/<REPO_NAME>/snapshots/aaaaaa/README.md
```

Diese `README.md`-Datei ist tatsächlich ein Symlink, der auf den Blob verweist, der den Hash der Datei hat.

Durch das Erstellen des Grundgerüsts auf diese Weise ermöglichen wir den Mechanismus der Dateifreigabe:
Wenn dieselbe Datei in der Revision `bbbbbb` abgerufen wurde, hätte sie denselben Hash und die Datei müsste nicht erneut heruntergeladen werden.

### .no_exist (fortgeschritten)

Zusätzlich zu den Ordnern `blobs`, `refs` und `snapshots` könnten Sie in Ihrem Cache auch einen `.no_exist` Ordner finden.
Dieser Ordner hält fest, welche Dateien Sie einmal versucht haben herunterzuladen, die jedoch nicht auf dem Hub vorhanden sind.
Seine Struktur ist dieselbe wie der `snapshots` Ordner mit einem Unterordner pro bekannter Revision:

```
<CACHE_DIR>/<REPO_NAME>/.no_exist/aaaaaa/config_that_does_not_exist.json
```

Im Gegensatz zum `snapshots` Ordner handelt es sich bei den Dateien um einfache leere Dateien (keine Symlinks).
In diesem Beispiel existiert die Datei `"config_that_does_not_exist.json"` nicht auf dem Hub für die Revision `"aaaaaa"`.
Da dieser Ordner nur leere Dateien speichert, ist sein Speicherplatzverbrauch vernachlässigbar.

Sie fragen sich jetzt vielleicht, warum diese Information überhaupt relevant ist?
In einigen Fällen versucht ein Framework, optionale Dateien für ein Modell zu laden.
Das Speichern der Nicht-Existenz optionaler Dateien beschleunigt das Laden eines Modells, da 1 HTTP-Anfrage pro möglicher optionaler Datei gespart wird.
Dies ist zum Beispiel bei `transformers` der Fall, wo jeder Tokenizer zusätzliche Dateien unterstützen kann. Beim ersten Laden des Tokenizers
auf Ihrem Gerät wird im Cache gespeichert, welche optionalen Dateien vorhanden sind (und welche nicht), um die Ladezeit bei den nächsten Initialisierungen zu beschleunigen.

Um zu testen, ob eine Datei lokal im Cache gespeichert ist (ohne eine HTTP-Anfrage zu senden), können Sie die [`try_to_load_from_cache`] Hilfsfunktion verwenden.
Sie gibt entweder den Dateipfad zurück (falls vorhanden und im Cache gespeichert), das Objekt `_CACHED_NO_EXIST` (wenn die Nicht-Existenz im Cache gespeichert ist)
oder `None` (wenn wir es nicht wissen).

```python
from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

filepath = try_to_load_from_cache()
if isinstance(filepath, str):
    # file exists and is cached
    ...
elif filepath is _CACHED_NO_EXIST:
    # non-existence of file is cached
    ...
else:
    # file is not cached
    ...
```

### In der Praxis

In der Praxis sollte Ihr Cache folgendermaßen aussehen:

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

### Einschränkungen

Um ein effizientes Cache-System zu haben, verwendet `huggingface-hub` Symlinks. Allerdings
werden Symlinks nicht auf allen Maschinen unterstützt. Dies ist eine bekannte Einschränkung,
insbesondere bei Windows. Wenn dies der Fall ist, verwendet `huggingface_hub` nicht das `blobs/` Verzeichnis,
sondern speichert die Dateien direkt im `snapshots/` Verzeichnis. Dieser Workaround ermöglicht es den Nutzern,
Dateien vom Hub auf genau die gleiche Weise herunterzuladen und zu cachen.
Auch Werkzeuge zur Überprüfung und Löschung des Caches (siehe unten) werden unterstützt.
Allerdings ist das Cache-System weniger effizient, da eine einzelne Datei möglicherweise mehrmals heruntergeladen wird,
wenn mehrere Revisionen des gleichen Repos heruntergeladen werden.

Wenn Sie von dem Symlink-basierten Cache-System auf einem Windows-Gerät profitieren möchten,
müssen Sie entweder den [Entwicklermodus aktivieren](https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development)
oder Python als Administrator ausführen.

Wenn Symlinks nicht unterstützt werden, wird dem Nutzer eine Warnmeldung angezeigt, um ihn darauf hinzuweisen,
dass er eine eingeschränkte Version des Cache-Systems verwendet. Diese Warnung kann durch Setzen der
Umgebungsvariable `HF_HUB_DISABLE_SYMLINKS_WARNING` auf true deaktiviert werden.

## Assets zwischenspeichern

Zusätzlich zum Zwischenspeichern von Dateien aus dem Hub benötigen nachgelagerte Bibliotheken
oft das Zwischenspeichern von anderen Dateien, die in Verbindung mit HF stehen, aber nicht
direkt von `huggingface_hub` behandelt werden (zum Beispiel: Dateien, die von GitHub heruntergeladen werden,
vorverarbeitete Daten, Protokolle,...). Um diese Dateien, die als `assets` bezeichnet werden, zwischenzuspeichern,
kann man [`cached_assets_path`] verwenden. Dieser kleine Helfer generiert Pfade im HF-Cache auf eine einheitliche Weise,
basierend auf dem Namen der anfragenden Bibliothek und optional auf einem Namensraum und einem Unterordnernamen.
Das Ziel ist, dass jede nachgelagerte Bibliothek ihre Assets auf ihre eigene Weise verwaltet
(z.B. keine Regelung über die Struktur), solange sie im richtigen Assets-Ordner bleibt.
Diese Bibliotheken können dann die Werkzeuge von `huggingface_hub` nutzen, um den Cache zu verwalten,
insbesondere um Teile der Assets über einen CLI-Befehl zu scannen und zu löschen.

```py
from huggingface_hub import cached_assets_path

assets_path = cached_assets_path(library_name="datasets", namespace="SQuAD", subfolder="download")
something_path = assets_path / "something.json" # Machen Sie, was Sie möchten, in Ihrem Assets-Ordner!
```

> [!TIP]
> [`cached_assets_path`] ist der empfohlene Weg, um Assets zu speichern, ist jedoch nicht verpflichtend.
> Wenn Ihre Bibliothek bereits ihren eigenen Cache verwendet, können Sie diesen gerne nutzen!

### Assets in der Praxis

In der Praxis sollte Ihr Assets-Cache wie der folgende Verzeichnisbaum aussehen:

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

## Cache scannen

Derzeit werden zwischengespeicherte Dateien nie aus Ihrem lokalen Verzeichnis gelöscht:
Wenn Sie eine neue Revision eines Zweiges herunterladen, werden vorherige Dateien aufbewahrt,
falls Sie sie wieder benötigen. Daher kann es nützlich sein, Ihr Cache-Verzeichnis zu scannen,
um zu erfahren, welche Repos und Revisionen den meisten Speicherplatz beanspruchen.
`huggingface_hub` bietet einen Helfer dafür, der über `hf` oder in einem Python-Skript verwendet werden kann.

### Cache im Terminal prüfen

Die bequemste Möglichkeit, Ihren HF-Cache zu untersuchen, ist der Befehl `hf cache ls`.
Er listet standardmäßig alle gecachten Repositories zusammen mit Größe, letzter Nutzung und Referenzen auf.

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

Mit `--revisions` wechseln Sie zur Ansicht auf Snapshot-Ebene. Filter akzeptieren
menschenlesbare Werte, sodass Ausdrücke wie `size>1GB` oder `accessed>30d` sofort funktionieren:

```text
➜ hf cache ls --revisions --filter "size>1GB" --filter "accessed>30d"
ID                                   REVISION            SIZE   LAST_MODIFIED REFS
------------------------------------ ------------------ ------- ------------- -------------------
model/bert-base-cased                6d1d7a1a2a6cf4c2    1.9G  2 years ago
model/t5-small                       1c610f6b3f5e7d8a    1.1G  3 months ago  main

Found 2 repo(s) for a total of 2 revision(s) and 3.0G on disk.
```

Brauchen Sie maschinenlesbare Ausgaben? `--format json` liefert strukturierte Objekte,
`--format csv` erzeugt durch Komma getrennte Zeilen und `--quiet` gibt nur Kennungen aus.
Alle Varianten lassen sich mit `--cache-dir` kombinieren, wenn Ihr Cache nicht unter `HF_HOME`
liegt.

#### Mit Shell-Tools filtern

Die Tabellen-Ausgabe lässt sich weiterhin mit bekannten Tools verarbeiten. Das folgende
Beispiel zeigt alle Revisionen für `t5-small`:

```text
➜ eval "hf cache ls --revisions" | grep "t5-small"
model/t5-small                       1c610f6b3f5e7d8a    1.1G  3 months ago  main
model/t5-small                       8f3ad1c90fed7a62    820.1M 2 weeks ago   refs/pr/1
```

### Den Cache von Python aus scannen

Für eine erweiterte Nutzung verwenden Sie [`scan_cache_dir`], welches das von dem CLI-Tool
aufgerufene Python-Dienstprogramm ist.

Sie können es verwenden, um einen detaillierten Bericht zu erhalten, der um 4 Datenklassen herum strukturiert ist:

- [`HFCacheInfo`]: vollständiger Bericht, der von [`scan_cache_dir`] zurückgegeben wird
- [`CachedRepoInfo`]: Informationen über ein gecachtes Repo
- [`CachedRevisionInfo`]: Informationen über eine gecachtes Revision (z.B. "snapshot) in einem Repo
- [`CachedFileInfo`]: Informationen über eine gecachte Datei in einem Snapshot

Hier ist ein einfaches Anwendungs-Beispiel in Python. Siehe Referenz für Details.

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
                    # No `last_accessed` as blobs are shared among revisions
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

## Cache leeren

Das Durchsuchen Ihres Caches ist interessant, aber was Sie normalerweise als Nächstes tun möchten, ist
einige Teile zu löschen, um Speicherplatz freizugeben. Dies gelingt mit den CLI-Befehlen
`hf cache rm` und `hf cache prune`. Alternativ können Sie programmatisch den
[`~HFCacheInfo.delete_revisions`]-Helfer des zurückgegebenen [`HFCacheInfo`]-Objekts nutzen.

### Löschstrategie

Um einige Cache zu löschen, müssen Sie eine Liste von Revisionen übergeben, die gelöscht werden sollen. Das Tool wird
eine Strategie definieren, um den Speicherplatz auf der Grundlage dieser Liste freizugeben. Es gibt ein
[`DeleteCacheStrategy`] Objekt zurück, das beschreibt, welche Dateien und Ordner gelöscht werden. Die
[`DeleteCacheStrategy`] zeigt Ihnen, wie viel Speicherplatz voraussichtlich frei wird.
Sobald Sie mit der Löschung einverstanden sind, müssen Sie sie ausführen, um die Löschung wirksam zu machen.
Um Abweichungen zu vermeiden, können Sie ein Strategieobjekt nicht manuell bearbeiten.

Die Strategie zur Löschung von Revisionen ist folgende:

- Der Ordner `snapshot`, der die Revisions-Symlinks enthält, wird gelöscht.
- Blob-Dateien, die nur von zu löschenden Revisionen verlinkt werden, werden ebenfalls gelöscht.
- Wenn eine Revision mit 1 oder mehreren `refs` verknüpft ist, werden die Referenzen gelöscht.
- Werden alle Revisionen aus einem Repo gelöscht, wird das gesamte zwischengespeicherte Repository gelöscht.

> [!TIP]
> Revisions-Hashes sind eindeutig über alle Repositories hinweg. `hf cache rm` akzeptiert daher sowohl
> Repository-Kennungen (z. B. `model/bert-base-uncased`) als auch einzelne Revisions-Hashes – bei einem Hash
> müssen Sie das Repository nicht zusätzlich angeben.

> [!WARNING]
> Wenn eine Revision im Cache nicht gefunden wird, wird sie stillschweigend ignoriert. Außerdem wird, wenn eine Datei
> oder ein Ordner beim Versuch, ihn zu löschen, nicht gefunden wird, eine Warnung protokolliert, aber es wird kein
> Fehler ausgelöst. Die Löschung wird für andere Pfade im
> [`DeleteCacheStrategy`] Objekt fortgesetzt.

### Cache vom Terminal aus leeren

Verwenden Sie `hf cache rm`, um gecachte Repositories oder einzelne Revisionen zu löschen.
Übergeben Sie dazu eine oder mehrere Repository-Kennungen (z. B. `model/bert-base-uncased`) oder Revisions-Hashes:

```text
➜ hf cache rm model/bert-base-cased
About to delete 1 repo(s) totalling 1.9G.
  - model/bert-base-cased (entire repo)
Proceed with deletion? [y/N]: y
Deleted 1 repo(s) and 1 revision(s); freed 1.9G.
```

Sie können Repositories und spezifische Revisionen mischen. Nutzen Sie `--dry-run`, um den Effekt vorab zu prüfen,
oder `--yes`, wenn keine Rückfrage erscheinen soll:

```text
➜ hf cache rm model/t5-small 8f3ad1c --dry-run
About to delete 1 repo(s) and 1 revision(s) totalling 1.1G.
  - model/t5-small:
      8f3ad1c [main] 1.1G
Dry run: no files were deleted.
```

Wenn Ihr Cache nicht im Standardverzeichnis liegt, kombinieren Sie den Befehl mit `--cache-dir PFAD`.

Zum Aufräumen verwaister Snapshots steht `hf cache prune` bereit. Der Befehl entfernt automatisch alle
Revisionen ohne Referenz:

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

Beide Befehle unterstützen `--dry-run`, `--yes` und `--cache-dir`, sodass Sie Vorschauen erzeugen,
Automatisierungen bauen und alternative Cache-Verzeichnisse angeben können.

### Cache aus Python leeren

Für mehr Flexibilität können Sie auch die Methode [`~HFCacheInfo.delete_revisions`] programmatisch verwenden.
Hier ist ein einfaches Beispiel. Siehe Referenz für Details.

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
