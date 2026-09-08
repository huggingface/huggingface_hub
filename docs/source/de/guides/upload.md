<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Dateien auf den Hub hochladen

Das Teilen Ihrer Dateien und Arbeiten ist ein wichtiger Aspekt des Hubs. Das `huggingface_hub` bietet mehrere Optionen, um Ihre Dateien auf den Hub hochzuladen. Sie können diese Funktionen unabhängig verwenden oder sie in Ihre Bibliothek integrieren, um es Ihren Benutzern zu erleichtern, mit dem Hub zu interagieren.

Wenn Sie Dateien auf den Hub hochladen möchten, müssen Sie sich bei Ihrem Hugging Face-Konto anmelden:

- Melden Sie sich bei Ihrem Hugging Face-Konto mit dem folgenden Befehl an:

  ```bash
  hf auth login
  # oder mit einer Umgebungsvariable
  hf auth login --token $HUGGINGFACE_TOKEN
  ```

- Alternativ können Sie sich in einem Notebook oder einem Skript programmatisch mit [`login`] anmelden:

  ```python
  >>> from huggingface_hub import login
  >>> login()
  ```

  Wenn es in einem Jupyter- oder Colaboratory-Notebook ausgeführt wird, startet [`login`] ein Widget, über das Sie Ihren Hugging Face-Zugriffstoken eingeben können. Andernfalls wird eine Meldung im Terminal angezeigt.

  Es ist auch möglich, sich programmatisch ohne das Widget anzumelden, indem Sie den Token direkt an [`login`] übergeben. Seien Sie jedoch vorsichtig, wenn Sie Ihr Notebook teilen. Es ist am Besten, den Token aus einem sicheren Passwortspeicher zu laden, anstatt ihn in Ihrem Colaboratory-Notebook zu speichern.

## Datei hochladen

Sobald Sie ein Repository mit [`create_repo`] erstellt haben, können Sie mit [`upload_file`] eine Datei in Ihr Repository hochladen.

Geben Sie den Pfad der hochzuladenden Datei, den Ort, an den Sie die Datei im Repository hochladen möchten, und den Namen des Repositories an, zu dem Sie die Datei hinzufügen möchten. Abhängig von Ihrem Repository-Typ können Sie optional den Repository-Typ als `dataset`, `model`, oder `space` festlegen.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.upload_file(
...     path_or_fileobj="/path/to/local/folder/README.md",
...     path_in_repo="README.md",
...     repo_id="username/test-dataset",
...     repo_type="dataset",
... )
```

## Ordner hochladen

Verwenden Sie die Funktion [`upload_folder`], um einen lokalen Ordner in ein vorhandenes Repository hochzuladen. Geben Sie den Pfad des lokalen Ordners an, den Sie hochladen möchten, an welchem Ort Sie den Ordner im Repository hochladen möchten, und den Namen des Repositories, zu dem Sie den Ordner hinzufügen möchten. Abhängig von Ihrem Repository-Typ können Sie optional den Repository-Typ als `dataset`, `model`, oder `space` festlegen.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()

# Den gesamten Inhalt aus dem lokalen Ordner in den entfernten Space hoch laden.
# Standardmäßig werden Dateien im Hauptverzeichnis des Repos hochgeladen
>>> api.upload_folder(
...     folder_path="/path/to/local/space",
...     repo_id="username/my-cool-space",
...     repo_type="space",
... )
```

Verwenden Sie die Argumente `allow_patterns` und `ignore_patterns`, um anzugeben, welche Dateien hochgeladen werden sollen. Diese Parameter akzeptieren entweder ein einzelnes Muster oder eine Liste von Mustern. Muster sind Standard-Wildcards (globbing patterns) wie [hier](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm) dokumentiert. Wenn sowohl `allow_patterns` als auch `ignore_patterns` angegeben werden, gelten beide Einschränkungen. Standardmäßig werden alle Dateien aus dem Ordner hochgeladen.

Jeder `.git/`-Ordner in einem Unterverzeichnis wird ignoriert. Bitte beachten Sie jedoch, dass die `.gitignore`-Datei nicht berücksichtigt wird. Dies bedeutet, dass Sie `allow_patterns` und `ignore_patterns` verwenden müssen, um anzugeben, welche Dateien stattdessen hochgeladen werden sollen.

```py
>>> api.upload_folder(
...     folder_path="/path/to/local/folder",
...     path_in_repo="my-dataset/train", # Hochladen in einen bestimmten Ordner
...     repo_id="username/test-dataset",
...     repo_type="dataset",
...     ignore_patterns="**/logs/*.txt", # Alle Textprotokolle ignorieren
... )
```

Sie können auch das Argument `delete_patterns` verwenden, um Dateien anzugeben, die Sie im selben Commit aus dem Repo löschen möchten. Dies kann nützlich sein, wenn Sie einen entfernten Ordner reinigen möchten, bevor Sie Dateien darin ablegen und nicht wissen, welche Dateien bereits vorhanden sind.

Im folgenden Beispiel wird der lokale Ordner `./logs` in den entfernten Ordner `/experiment/logs/` hochgeladen. Es werden nur txt-Dateien hochgeladen, aber davor werden alle vorherigen Protokolle im Repo gelöscht. All dies in einem einzigen Commit.

```py
>>> api.upload_folder(
...     folder_path="/path/to/local/folder/logs",
...     repo_id="username/trained-model",
...     path_in_repo="experiment/logs/",
...     allow_patterns="*.txt", # Alle lokalen Textdateien hochladen
...     delete_patterns="*.txt", # Vorher alle enfernten Textdateien löschen
... )
```

## Erweiterte Funktionen

In den meisten Fällen benötigen Sie nicht mehr als [`upload_file`] und [`upload_folder`], um Ihre Dateien auf den Hub hochzuladen. Das `huggingface_hub` bietet jedoch fortschrittlichere Funktionen, um die Dinge einfacher zu machen. Schauen wir sie uns an!


### Nicht blockierende Uploads

In einigen Fällen möchten Sie Daten hochladen, ohne Ihren Hauptthread zu blockieren. Dies ist besonders nützlich, um Protokolle und Artefakte hochzuladen, während Sie weiter trainieren. Um dies zu tun, können Sie das Argument `run_as_future` in beiden [`upload_file`] und [`upload_folder`] verwenden. Dies gibt ein [`concurrent.futures.Future`](https://docs.python.org/3/library/concurrent.futures.html#future-objects)-Objekt zurück, mit dem Sie den Status des Uploads überprüfen können.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> future = api.upload_folder( # Hochladen im Hintergrund (nicht blockierende Aktion)
...     repo_id="username/my-model",
...     folder_path="checkpoints-001",
...     run_as_future=True,
... )
>>> future
Future(...)
>>> future.done()
False
>>> future.result() # Warten bis der Upload abgeschlossen ist (blockierende Aktion)
...
```

> [!TIP]
> Hintergrund-Aufgaben werden in die Warteschlange gestellt, wenn `run_as_future=True` verwendet wird. Das bedeutet, dass garantiert wird, dass die Aufgaben in der richtigen Reihenfolge ausgeführt werden.

Auch wenn Hintergrundaufgaben hauptsächlich dazu dienen, Daten hochzuladen/Commits zu erstellen, können Sie jede gewünschte Methode in die Warteschlange stellen, indem Sie [`run_as_future`] verwenden. Sie können es beispielsweise verwenden, um ein Repo zu erstellen und dann Daten im Hintergrund dorthin hochzuladen. Das integrierte Argument `run_as_future` in Upload-Methoden ist lediglich ein Alias dafür.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.run_as_future(api.create_repo, "username/my-model", exists_ok=True)
Future(...)
>>> api.upload_file(
...     repo_id="username/my-model",
...     path_in_repo="file.txt",
...     path_or_fileobj=b"file content",
...     run_as_future=True,
... )
Future(...)
```

### Ordner in Teilen hochladen

Mit [`upload_folder`] können Sie ganz einfach einen gesamten Ordner ins Hub hochladen. Bei großen Ordnern (Tausende von Dateien oder Hunderte von GB) kann dies jedoch immer noch herausfordernd sein. Wenn Sie einen Ordner mit vielen Dateien haben, möchten Sie ihn möglicherweise in mehreren Commits hochladen. Wenn während des Uploads ein Fehler oder ein Verbindungsproblem auftritt, müssen Sie den Vorgang nicht von Anfang an wiederholen.

Um einen Ordner in mehreren Commits hochzuladen, übergeben Sie einfach `multi_commits=True` als Argument. Intern wird `huggingface_hub` die hochzuladenden/zu löschenden Dateien auflisten und sie in mehrere Commits aufteilen. Die "Strategie" (d.h. wie die Commits aufgeteilt werden) basiert auf der Anzahl und Größe der hochzuladenden Dateien. Ein PR wird im Hub geöffnet, um alle Commits zu pushen. Sobald der PR bereit ist, werden die Commits zu einem einzigen Commit zusammengefasst. Wenn der Prozess unterbrochen wird, bevor er abgeschlossen ist, können Sie Ihr Skript erneut ausführen, um den Upload fortzusetzen. Der erstellte PR wird automatisch erkannt und der Upload setzt dort fort, wo er gestoppt wurde. Es wird empfohlen, `multi_commits_verbose=True` zu übergeben, um ein besseres Verständnis für den Upload und dessen Fortschritt zu erhalten.

Das untenstehende Beispiel lädt den Ordner "checkpoints" in ein Dataset in mehreren Commits hoch. Ein PR wird im Hub erstellt und automatisch zusammengeführt, sobald der Upload abgeschlossen ist. Wenn Sie möchten, dass der PR offen bleibt und Sie ihn manuell überprüfen können, übergeben Sie `create_pr=True`.

```py
>>> upload_folder(
...     folder_path="local/checkpoints",
...     repo_id="username/my-dataset",
...     repo_type="dataset",
...     multi_commits=True,
...     multi_commits_verbose=True,
... )
```

Wenn Sie die Upload-Strategie besser steuern möchten (d.h. die erstellten Commits), können Sie sich die Low-Level-Methoden [`plan_multi_commits`] und [`create_commits_on_pr`] ansehen.

> [!WARNING]
> `multi_commits` ist noch ein experimentelles Feature. Seine API und sein Verhalten können in Zukunft ohne vorherige Ankündigung geändert werden.

### Geplante Uploads

Das Hugging Face Hub erleichtert das Speichern und Versionieren von Daten. Es gibt jedoch einige Einschränkungen, wenn Sie dieselbe Datei Tausende von Malen aktualisieren möchten. Sie möchten beispielsweise Protokolle eines Trainingsprozesses oder Benutzerfeedback in einem bereitgestellten Space speichern. In diesen Fällen macht es Sinn, die Daten als Dataset im Hub hochzuladen, aber es kann schwierig sein, dies richtig zu machen. Der Hauptgrund ist, dass Sie nicht jede Aktualisierung Ihrer Daten versionieren möchten, da dies das git-Repository unbrauchbar machen würde. Die Klasse [`CommitScheduler`] bietet eine Lösung für dieses Problem.

Die Idee besteht darin, einen Hintergrundjob auszuführen, der regelmäßig einen lokalen Ordner ins Hub schiebt. Nehmen Sie an, Sie haben einen Gradio Space, der als Eingabe einen Text nimmt und zwei Übersetzungen davon generiert. Der Benutzer kann dann seine bevorzugte Übersetzung auswählen. Für jeden Durchlauf möchten Sie die Eingabe, Ausgabe und Benutzerpräferenz speichern, um die Ergebnisse zu analysieren. Dies ist ein perfekter Anwendungsfall für [`CommitScheduler`]; Sie möchten Daten ins Hub speichern (potenziell Millionen von Benutzerfeedbacks), aber Sie müssen nicht in Echtzeit jede Benutzereingabe speichern. Stattdessen können Sie die Daten lokal in einer JSON-Datei speichern und sie alle 10 Minuten hochladen. Zum Beispiel:

```py
>>> import json
>>> import uuid
>>> from pathlib import Path
>>> import gradio as gr
>>> from huggingface_hub import CommitScheduler

# Definieren Sie die Datei, in der die Daten gespeichert werden sollen. Verwenden Sie UUID, um sicherzustellen, dass vorhandene Daten aus einem früheren Lauf nicht überschrieben werden.
>>> feedback_file = Path("user_feedback/") / f"data_{uuid.uuid4()}.json"
>>> feedback_folder = feedback_file.parent

# Geplante regelmäßige Uploads. Das Remote-Repo und der lokale Ordner werden erstellt, wenn sie noch nicht existieren.
>>> scheduler = CommitScheduler(
...     repo_id="report-translation-feedback",
...     repo_type="dataset",
...     folder_path=feedback_folder,
...     path_in_repo="data",
...     every=10,
... )

# Eine einfache Gradio-Anwendung, die einen Text als Eingabe nimmt und zwei Übersetzungen generiert. Der Benutzer wählt seine bevorzugte Übersetzung aus.
>>> def save_feedback(input_text:str, output_1: str, output_2:str, user_choice: int) -> None:
...     """
...     Füge Eingabe/Ausgabe und Benutzerfeedback zu einer JSON-Lines-Datei hinzu und verwende ein Thread-Lock, um gleichzeitiges Schreiben von verschiedenen Benutzern zu vermeiden.
...     """
...     with scheduler.lock:
...         with feedback_file.open("a") as f:
...             f.write(json.dumps({"input": input_text, "output_1": output_1, "output_2": output_2, "user_choice": user_choice}))
...             f.write("\n")

# Starte Gradio
>>> with gr.Blocks() as demo:
>>>     ... # Definiere Gradio Demo + verwende `save_feedback`
>>> demo.launch()
```

Und das war's! Benutzereingabe/-ausgaben und Feedback sind als Dataset auf dem Hub verfügbar. Durch die Verwendung eines eindeutigen JSON-Dateinamens können Sie sicher sein, dass Sie keine Daten von einem vorherigen Lauf oder Daten von anderen Spaces/Replikas überschreiben, die gleichzeitig in dasselbe Repository pushen.

Für weitere Details über den [`CommitScheduler`], hier das Wichtigste:

- **append-only / Nur hinzufügen:**
      Es wird davon ausgegangen, dass Sie nur Inhalte zum Ordner hinzufügen. Sie dürfen nur Daten zu bestehenden Dateien hinzufügen oder neue Dateien erstellen. Das Löschen oder Überschreiben einer Datei könnte Ihr Repository beschädigen.
- **git history / git Historie**:
      Der Scheduler wird den Ordner alle every Minuten committen. Um das Git-Repository nicht zu überladen, wird empfohlen, einen minimalen Wert von 5 Minuten festzulegen. Außerdem ist der Scheduler darauf ausgelegt, leere Commits zu vermeiden. Wenn im Ordner kein neuer Inhalt erkannt wird, wird der geplante Commit verworfen.
- **errors / Fehler**:
      Der Scheduler läuft als Hintergrund-Thread. Er wird gestartet, wenn Sie die Klasse instanziieren, und stoppt nie. Insbesondere, wenn während des Uploads ein Fehler auftritt (z. B. Verbindungsproblem), wird der Scheduler ihn stillschweigend ignorieren und beim nächsten geplanten Commit erneut versuchen.
- **thread-safety / Thread-Sicherheit**:
      In den meisten Fällen können Sie davon ausgehen, dass Sie eine Datei schreiben können, ohne sich um eine Lock-Datei kümmern zu müssen. Der Scheduler wird nicht abstürzen oder beschädigt werden, wenn Sie Inhalte in den Ordner schreiben, während er hochlädt. In der Praxis ist es möglich, dass bei stark ausgelasteten Apps Probleme mit der Parallelität auftreten. In diesem Fall empfehlen wir, das `scheduler.lock` Lock zu verwenden, um die Thread-Sicherheit zu gewährleisten. Das Lock wird nur gesperrt, wenn der Scheduler den Ordner auf Änderungen überprüft, nicht beim Hochladen von Daten. Sie können sicher davon ausgehen, dass dies das Benutzererlebnis in Ihrem Space nicht beeinflusst.

#### Space Persistenz-Demo

Das Speichern von Daten aus einem Space in einem Dataset auf dem Hub ist der Hauptanwendungsfall für den [`CommitScheduler`]. Je nach Anwendungsfall möchten Sie Ihre Daten möglicherweise anders strukturieren. Die Struktur muss robust gegenüber gleichzeitigen Benutzern und Neustarts sein, was oft das Generieren von UUIDs impliziert. Neben der Robustheit sollten Sie Daten in einem Format hochladen, das von der 🤗 Datasets-Bibliothek für die spätere Wiederverwendung gelesen werden kann. Wir haben einen [Space](https://huggingface.co/spaces/Wauplin/space_to_dataset_saver) erstellt, der zeigt, wie man verschiedene Datenformate speichert (dies muss möglicherweise für Ihre speziellen Bedürfnisse angepasst werden).

#### Benutzerdefinierte Uploads

[`CommitScheduler`] geht davon aus, dass Ihre Daten nur hinzugefügt werden und "wie sie sind" hochgeladen werden sollten. Sie möchten jedoch möglicherweise anpassen, wie Daten hochgeladen werden. Dies können Sie tun, indem Sie eine Klasse erstellen, die vom [`CommitScheduler`] erbt und die Methode `push_to_hub` überschreibt (fühlen Sie sich frei, sie nach Belieben zu überschreiben). Es ist garantiert, dass sie alle `every` Minuten in einem Hintergrund-Thread aufgerufen wird. Sie müssen sich keine Gedanken über Parallelität und Fehler machen, aber Sie müssen vorsichtig sein bei anderen Aspekten, wie z. B. dem Pushen von leeren Commits oder doppelten Daten.

Im folgenden (vereinfachten) Beispiel überschreiben wir `push_to_hub`, um alle PNG-Dateien in einem einzigen Archiv zu zippen, um das Repo auf dem Hub nicht zu überladen:

```py
class ZipScheduler(CommitScheduler):
    def push_to_hub(self):
        # 1. Liste PNG-Dateien auf
          png_files = list(self.folder_path.glob("*.png"))
          if len(png_files) == 0:
              return None  # kehre früh zurück, wenn nichts zu committen ist

        # 2. Zippe PNG-Dateien in ein einzelnes Archiv
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "train.zip"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zip:
                for png_file in png_files:
                    zip.write(filename=png_file, arcname=png_file.name)

            # 3. Lade das Archiv hoch
            self.api.upload_file(..., path_or_fileobj=archive_path)

        # 4. Lösche lokale PNG-Dateien, um späteres erneutes Hochladen zu vermeiden
        for png_file in png_files:
            png_file.unlink()
```

Wenn Sie `push_to_hub` überschreiben, haben Sie Zugriff auf die Attribute vom [`CommitScheduler`] und insbesondere:
- [`HfApi`] Client: `api`
- Ordnerparameter: `folder_path` und `path_in_repo`
- Repo-Parameter: `repo_id`, `repo_type`, `revision`
- Das Thread-Lock: `lock`

> [!TIP]
> Für weitere Beispiele von benutzerdefinierten Schedulern, schauen Sie sich unseren [Demo Space](https://huggingface.co/spaces/Wauplin/space_to_dataset_saver) an, der verschiedene Implementierungen je nach Ihren Anforderungen enthält.

### create_commit

Die Funktionen [`upload_file`] und [`upload_folder`] sind High-Level-APIs, die im Allgemeinen bequem zu verwenden sind. Wir empfehlen, diese Funktionen zuerst auszuprobieren, wenn Sie nicht auf einer niedrigeren Ebene arbeiten müssen. Wenn Sie jedoch auf Commit-Ebene arbeiten möchten, können Sie die Funktion [`create_commit`] direkt verwenden.

Es gibt drei von [`create_commit`] unterstützte Operationstypen:

- [`CommitOperationAdd`] lädt eine Datei in den Hub hoch. Wenn die Datei bereits existiert, werden die Dateiinhalte überschrieben. Diese Operation akzeptiert zwei Argumente:

  - `path_in_repo`: der Repository-Pfad, um eine Datei hochzuladen.
  - `path_or_fileobj`: entweder ein Pfad zu einer Datei auf Ihrem Dateisystem oder ein Datei-ähnliches Objekt. Dies ist der Inhalt der Datei, die auf den Hub hochgeladen werden soll.
- [`CommitOperationDelete`] entfernt eine Datei oder einen Ordner aus einem Repository. Diese Operation akzeptiert path_in_repo als Argument.
- [`CommitOperationCopy`] kopiert eine Datei innerhalb eines Repositorys. Diese Operation akzeptiert drei Argumente:
  - `src_path_in_repo`: der Repository-Pfad der zu kopierenden Datei.
  - `path_in_repo`: der Repository-Pfad, wohin die Datei kopiert werden soll.
  - `src_revision`: optional - die Revision der zu kopierenden Datei, wenn Sie eine Datei von einem anderen Branch/Revision kopieren möchten.

Zum Beispiel, wenn Sie zwei Dateien hochladen und eine Datei in einem Hub-Repository löschen möchten:

1. Verwenden Sie die entsprechende `CommitOperation`, um eine Datei hinzuzufügen oder zu löschen und einen Ordner zu löschen:

```py
>>> from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
>>> api = HfApi()
>>> operations = [
...     CommitOperationAdd(path_in_repo="LICENSE.md", path_or_fileobj="~/repo/LICENSE.md"),
...     CommitOperationAdd(path_in_repo="weights.h5", path_or_fileobj="~/repo/weights-final.h5"),
...     CommitOperationDelete(path_in_repo="old-weights.h5"),
...     CommitOperationDelete(path_in_repo="logs/"),
...     CommitOperationCopy(src_path_in_repo="image.png", path_in_repo="duplicate_image.png"),
... ]
```

2. Übergeben Sie Ihre Operationen an [`create_commit`]:

```py
>>> api.create_commit(
...     repo_id="lysandre/test-model",
...     operations=operations,
...     commit_message="Hochladen meiner Modell-Gewichte und -Lizenz",
... )
```

Zusätzlich zu [`upload_file`] und [`upload_folder`] verwenden auch die folgenden Funktionen [`create_commit`] im Hintergrund:

- [`delete_file`] löscht eine einzelne Datei aus einem Repository auf dem Hub.
- [`delete_folder`] löscht einen gesamten Ordner aus einem Repository auf dem Hub.
- [`metadata_update`] aktualisiert die Metadaten eines Repositorys.

Für detailliertere Informationen werfen Sie einen Blick auf die [`HfApi`] Referenz.

## Tipps und Tricks für große Uploads

Bei der Verwaltung einer großen Datenmenge in Ihrem Repo gibt es einige Einschränkungen zu beachten. Angesichts der Zeit, die es dauert, die Daten zu streamen, kann es sehr ärgerlich sein, am Ende des Prozesses einen Upload/Push zu verlieren oder eine degradierte Erfahrung zu machen, sei es auf hf.co oder bei lokalem Arbeiten. Wir haben eine Liste von Tipps und Empfehlungen zusammengestellt, um Ihr Repo zu strukturieren.

| Eigenschaft     | Empfohlen        | Tipps                                                  |
| ----------------   | ------------------ | ------------------------------------------------------ |
| Repo-Größe         | -                  | Kontaktieren Sie uns für große Repos (TBs Daten)               |
| Dateien pro Repo     | <100k              | Daten in weniger Dateien zusammenführen                            |
| Einträge pro Ordner | <10k               | Unterverzeichnisse im Repo verwenden                             |
| Dateigröße         | <5GB               | Daten in geteilte Dateien aufteilen                          |
| Commit-Größe        | <100 files*        | Dateien in mehreren Commits hochladen                       |
| Commits pro Repo   | -                  | Mehrere Dateien pro Commit hochladen und/oder Historie zusammenführen |

_* Nicht relevant bei direkter Verwendung des `git` CLI_

Bitte lesen Sie den nächsten Abschnitt, um diese Beschränkungen besser zu verstehen und zu erfahren, wie Sie damit umgehen können.

### Hub-Repository Größenbeschränkungen

Was meinen wir, wenn wir von "großen Uploads" sprechen, und welche Einschränkungen sind damit verbunden? Große Uploads können sehr unterschiedlich sein, von Repositories mit einigen riesigen Dateien (z. B. Modellgewichten) bis hin zu Repositories mit Tausenden von kleinen Dateien (z. B. einem Bilddatensatz).

Hinter den Kulissen verwendet der Hub Git zur Versionierung der Daten, was strukturelle Auswirkungen darauf hat, was Sie in Ihrem Repo tun können.
Wenn Ihr Repo einige der im vorherigen Abschnitt erwähnten Zahlen überschreitet, **empfehlen wir Ihnen dringend, [`git-sizer`](https://github.com/github/git-sizer) zu verwenden**, das eine sehr detaillierte Dokumentation über die verschiedenen Faktoren bietet, die Ihr Erlebnis beeinflussen werden. Hier ist ein TL;DR der zu berücksichtigenden Faktoren:

- **Repository-Größe**: Die Gesamtgröße der Daten, die Sie hochladen möchten. Es gibt keine feste Obergrenze für die Größe eines Hub-Repositories. Wenn Sie jedoch vorhaben, Hunderte von GBs oder sogar TBs an Daten hochzuladen, würden wir es begrüßen, wenn Sie uns dies im Voraus mitteilen könnten, damit wir Ihnen besser helfen können, falls Sie während des Prozesses Fragen haben. Sie können uns unter datasets@huggingface.co oder auf [unserem Discord](http://hf.co/join/discord) kontaktieren.
- **Anzahl der Dateien**:
    - Für ein optimales Erlebnis empfehlen wir, die Gesamtzahl der Dateien unter 100k zu halten. Versuchen Sie, die Daten zu weniger Dateien zusammenzuführen, wenn Sie mehr haben.
      Zum Beispiel können json-Dateien zu einer einzigen jsonl-Datei zusammengeführt oder große Datensätze als Parquet-Dateien exportiert werden.
    - Die maximale Anzahl von Dateien pro Ordner darf 10k Dateien pro Ordner nicht überschreiten. Eine einfache Lösung besteht darin, eine Repository-Struktur zu erstellen, die Unterverzeichnisse verwendet. Ein Repo mit 1k Ordnern von `000/` bis `999/`, in dem jeweils maximal 1000 Dateien enthalten sind, reicht bereits aus.
- **Dateigröße**: Bei hochzuladenden großen Dateien (z. B. Modellgewichte) empfehlen wir dringend, sie **in Blöcke von etwa 5GB aufzuteilen**.
Es gibt mehrere Gründe dafür:
    - Das Hoch- und Herunterladen kleinerer Dateien ist sowohl für Sie als auch für andere Benutzer viel einfacher. Bei der Datenübertragung können immer Verbindungsprobleme auftreten, und kleinere Dateien vermeiden das erneute Starten von Anfang an im Falle von Fehlern.
    - Dateien werden den Benutzern über CloudFront bereitgestellt. Aus unserer Erfahrung werden riesige Dateien von diesem Dienst nicht zwischengespeichert, was zu einer langsameren Downloadgeschwindigkeit führt.
In jedem Fall wird keine einzelne LFS-Datei >50GB sein können. D. h. 50GB ist das absolute Limit für die Einzeldateigröße.
- **Anzahl der Commits**: Es gibt kein festes Limit für die Gesamtzahl der Commits in Ihrer Repo-Historie. Aus unserer Erfahrung heraus beginnt das Benutzererlebnis im Hub jedoch nach einigen Tausend Commits abzunehmen. Wir arbeiten ständig daran, den Service zu verbessern, aber man sollte immer daran denken, dass ein Git-Repository nicht als Datenbank mit vielen Schreibzugriffen gedacht ist. Wenn die Historie Ihres Repos sehr groß wird, können Sie immer alle Commits mit [`super_squash_history`] zusammenfassen, um einen Neuanfang zu erhalten. Dies ist eine nicht rückgängig zu machende Operation.
- **Anzahl der Operationen pro Commit**: Auch hier gibt es keine feste Obergrenze. Wenn ein Commit im Hub hochgeladen wird, wird jede Git-Operation (Hinzufügen oder Löschen) vom Server überprüft. Wenn hundert LFS-Dateien auf einmal committed werden, wird jede Datei einzeln überprüft, um sicherzustellen, dass sie korrekt hochgeladen wurde. Beim Pushen von Daten über HTTP mit `huggingface_hub` wird ein Timeout von 60s für die Anforderung festgelegt, was bedeutet, dass, wenn der Prozess mehr Zeit in Anspruch nimmt, clientseitig ein Fehler ausgelöst wird. Es kann jedoch (in seltenen Fällen) vorkommen, dass selbst wenn das Timeout clientseitig ausgelöst wird, der Prozess serverseitig dennoch abgeschlossen wird. Dies kann manuell überprüft werden, indem man das Repo im Hub durchsucht. Um dieses Timeout zu vermeiden, empfehlen wir, pro Commit etwa 50-100 Dateien hinzuzufügen.

### Praktische Tipps

Nachdem wir die technischen Aspekte gesehen haben, die Sie bei der Strukturierung Ihres Repositories berücksichtigen müssen, schauen wir uns einige praktische Tipps an, um Ihren Upload-Prozess so reibungslos wie möglich zu gestalten.

- **Fangen Sie klein an**: Wir empfehlen, mit einer kleinen Datenmenge zu beginnen, um Ihr Upload-Skript zu testen. Es ist einfacher, an einem Skript zu arbeiten, wenn ein Fehler nur wenig Zeit kostet.
- **Rechnen Sie mit Ausfällen**: Das Streamen großer Datenmengen ist eine Herausforderung. Sie wissen nicht, was passieren kann, aber es ist immer am besten anzunehmen, dass etwas mindestens einmal schiefgehen wird - unabhängig davon, ob es an Ihrem Gerät, Ihrer Verbindung oder unseren Servern liegt. Wenn Sie zum Beispiel vorhaben, eine große Anzahl von Dateien hochzuladen, ist es am besten, lokal zu verfolgen, welche Dateien Sie bereits hochgeladen haben, bevor Sie die nächste Batch hochladen. Sie können sicher sein, dass eine LFS-Datei, die bereits committed wurde, niemals zweimal hochgeladen wird, aber es kann clientseitig trotzdem Zeit sparen, dies zu überprüfen.
- **~~Verwenden Sie `hf_transfer`~~** (deprecated): `hf_transfer` wird nicht mehr verwendet. Der Hugging Face Hub nutzt jetzt das Xet-Speicher-Backend, und alle Dateiübertragungen erfolgen über `hf-xet`. Für Hochleistungsübertragungen setzen Sie die Umgebungsvariable `HF_XET_HIGH_PERFORMANCE=1`. Weitere Informationen finden Sie unter [`HF_XET_HIGH_PERFORMANCE`](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfxethighperformance).
