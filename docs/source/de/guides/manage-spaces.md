<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Verwalten Ihres Spaces (Bereiches)

In diesem Leitfaden werden wir sehen, wie man den Laufzeitbereich eines Space
([Geheimnisse (Secrets)](https://huggingface.co/docs/hub/spaces-overview#managing-secrets),
[Hardware](https://huggingface.co/docs/hub/spaces-gpus) und Speicher (Storage)) mit `huggingface_hub` verwaltet.

## Ein einfaches Beispiel: Konfigurieren von Geheimnissen und Hardware

Hier ist ein End-to-End-Beispiel, um einen Space auf dem Hub zu erstellen und einzurichten.

**1. Einen Space auf dem Hub erstellen.**

```py
>>> from huggingface_hub import HfApi
>>> repo_id = "Wauplin/my-cool-training-space"
>>> api = HfApi()

# Zum Beispiel mit einem Gradio SDK
>>> api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio")
```

**1. (bis) Duplizieren eines Space.**

Das kann nützlich sein, wenn Sie auf einem bestehenden Space aufbauen möchten, anstatt von Grund auf neu zu beginnen.
Es ist auch nützlich, wenn Sie die Kontrolle über die Konfiguration/Einstellungen eines öffentlichen Space haben möchten. Siehe [`duplicate_space`] für weitere Details.

```py
>>> api.duplicate_space("multimodalart/dreambooth-training")
```

**2. Code mit bevorzugter Lösung hochladen.**

Hier ist ein Beispiel, wie man den lokalen Ordner `src/` von Ihrem Computer in Ihren Space hochlädt:

```py
>>> api.upload_folder(repo_id=repo_id, repo_type="space", folder_path="src/")
```

In diesem Schritt sollte Ihre App bereits kostenlos auf dem Hub laufen!
Möglicherweise möchten Sie sie jedoch weiterhin mit Geheimnissen und aufgerüsteter Hardware konfigurieren.

**3. Konfigurieren von Geheimnissen und Variablen**

Ihr Space könnte einige geheime Schlüssel, Tokens oder Variablen benötigen, um zu funktionieren.
Siehe [Dokumentation](https://huggingface.co/docs/hub/spaces-overview#managing-secrets) für weitere Details.
Zum Beispiel ein HF-Token, um einen Bilddatensatz auf den Hub hochzuladen, sobald er aus Ihrem Space generiert wurde.

```py
>>> api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value="hf_api_***")
>>> api.add_space_variable(repo_id=repo_id, key="MODEL_REPO_ID", value="user/repo")
```

Geheimnisse und Variablen können auch gelöscht werden:
```py
>>> api.delete_space_secret(repo_id=repo_id, key="HF_TOKEN")
>>> api.delete_space_variable(repo_id=repo_id, key="MODEL_REPO_ID")
```

> [!TIP]
> Innerhalb Ihres Space sind Geheimnisse als Umgebungsvariablen verfügbar (oder
> Streamlit Secrets Management, wenn Streamlit verwendet wird). Keine Notwendigkeit, sie über die API abzurufen!

> [!WARNING]
> Jede Änderung in der Konfiguration Ihres Space (Geheimnisse oder Hardware) wird einen Neustart Ihrer App auslösen.

**Bonus: Geheimnisse und Variablen beim Erstellen oder Duplizieren des Space festlegen!**

Geheimnisse und Variablen können beim Erstellen oder Duplizieren eines Space gesetzt werden:

```py
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio",
...     space_secrets=[{"key"="HF_TOKEN", "value"="hf_api_***"}, ...],
...     space_variables=[{"key"="MODEL_REPO_ID", "value"="user/repo"}, ...],
... )
```

```py
>>> api.duplicate_space(
...     from_id=repo_id,
...     secrets=[{"key"="HF_TOKEN", "value"="hf_api_***"}, ...],
...     variables=[{"key"="MODEL_REPO_ID", "value"="user/repo"}, ...],
... )
```

**4. Konfigurieren von Hardware**

Standardmäßig wird Ihr Space kostenlos in einer CPU-Umgebung ausgeführt. Sie können die Hardware
aktualisieren, um sie auf GPUs laufen zu lassen. Eine Zahlungskarte oder ein Community-Grant wird benötigt, um Ihren
Space zu aktualisieren. Siehe [Dokumentation](https://huggingface.co/docs/hub/spaces-gpus) für weitere Details.

```py
# Verwenden von `SpaceHardware` Enum
>>> from huggingface_hub import SpaceHardware
>>> api.request_space_hardware(repo_id=repo_id, hardware=SpaceHardware.T4_MEDIUM)

# Oder einfach einen String-Wert angeben
>>> api.request_space_hardware(repo_id=repo_id, hardware="t4-medium")
```

Hardware-Aktualisierungen erfolgen nicht sofort, da Ihr Space auf unseren Servern neu geladen werden muss.
Jederzeit können Sie überprüfen, auf welcher Hardware Ihr Space läuft, um zu sehen, ob Ihre Anfrage
erfüllt wurde.

```py
>>> runtime = api.get_space_runtime(repo_id=repo_id)
>>> runtime.stage
"RUNNING_BUILDING"
>>> runtime.hardware
"cpu-basic"
>>> runtime.requested_hardware
"t4-medium"
```

Sie verfügen jetzt über einen vollständig konfigurierten Space. Stellen Sie sicher, dass Sie Ihren Space wieder auf "cpu-classic"
zurückstufen, wenn Sie ihn nicht mehr verwenden.

**Bonus: Hardware beim Erstellen oder Duplizieren des Space anfordern!**

Aktualisierte Hardware wird Ihrem Space automatisch zugewiesen, sobald er erstellt wurde.

```py
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio"
...     space_hardware="cpu-upgrade",
...     space_storage="small",
...     space_sleep_time="7200", # 2 hours in secs
... )
```

```py
>>> api.duplicate_space(
...     from_id=repo_id,
...     hardware="cpu-upgrade",
...     storage="small",
...     sleep_time="7200", # 2 hours in secs
... )
```

**5. Pausieren und Neustarten des Spaces**

Standardmäßig, wenn Ihr Space auf augewerteter Hardware läuft, wird er nie angehalten. Um jedoch zu vermeiden, dass Ihnen Gebühren berechnet werden,
möchten Sie ihn möglicherweise anhalten, wenn Sie ihn nicht verwenden. Dies ist mit [`pause_space`] möglich. Ein pausierter Space bleibt
inaktiv, bis der Besitzer des Space ihn entweder über die Benutzeroberfläche oder über die API mit [`restart_space`] neu startet.
Weitere Informationen zum Pausenmodus finden Sie in [diesem Abschnitt](https://huggingface.co/docs/hub/spaces-gpus#pause).

```py
# Pausieren des Space, um Gebühren zu vermeiden
>>> api.pause_space(repo_id=repo_id)
# (...)
# Erneut starten, wenn benötigt
>>> api.restart_space(repo_id=repo_id)
```

Eine weitere Möglichkeit besteht darin, für Ihren Space einen Timeout festzulegen. Wenn Ihr Space länger als die Timeout-Dauer inaktiv ist,
wird er in den Schlafmodus versetzt. Jeder Besucher, der auf Ihren Space zugreift, wird ihn wieder starten. Sie können ein Timeout mit
[`set_space_sleep_time`] festlegen. Weitere Informationen zum Schlafmodus finden Sie in [diesem Abschnitt](https://huggingface.co/docs/hub/spaces-gpus#sleep-time).

```py
# Setzen den Space nach 1h Inaktivität in den Schlafmodus
>>> api.set_space_sleep_time(repo_id=repo_id, sleep_time=3600)
```

Hinweis: Wenn Sie eine 'cpu-basic' Hardware verwenden, können Sie keine benutzerdefinierte Schlafzeit konfigurieren. Ihr Space wird automatisch
nach 48h Inaktivität pausiert.

**Bonus: Schlafzeit festlegen, während der Hardwareanforderung**

Aufgewertete Hardware wird Ihrem Space automatisch zugewiesen, sobald er erstellt wurde.

```py
>>> api.request_space_hardware(repo_id=repo_id, hardware=SpaceHardware.T4_MEDIUM, sleep_time=3600)
```

**Bonus: Schlafzeit beim Erstellen oder Duplizieren des Space festlegen!**

```py
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio"
...     space_hardware="t4-medium",
...     space_sleep_time="3600",
... )
```

```py
>>> api.duplicate_space(
...     from_id=repo_id,
...     hardware="t4-medium",
...     sleep_time="3600",
... )
```

**6. Dem Space dauerhaften Speicherplatz hinzufügen**

Sie können den Speicher-Tier Ihrer Wahl auswählen, um auf Festplattenspeicher zuzugreifen, der Neustarts Ihres Space überdauert. Dies bedeutet, dass Sie von der Festplatte lesen und darauf schreiben können, wie Sie es von einer herkömmlichen Festplatte gewöhnt sind. Weitere Informationen finden Sie in der [Dokumentation](https://huggingface.co/docs/hub/spaces-storage#persistent-storage) .

```py
>>> from huggingface_hub import SpaceStorage
>>> api.request_space_storage(repo_id=repo_id, storage=SpaceStorage.LARGE)
```

Sie können auch Ihren Speicher löschen und dabei alle Daten dauerhaft verlieren.
```py
>>> api.delete_space_storage(repo_id=repo_id)
```

Hinweis: Nachdem Ihnen ein Speicher-Tier zugewiesen wurde, können Sie diesen nicht mehr herabsetzen. Um dies zu tun, müssen Sie zuerst den Speicher löschen und dann den gewünschten Tier anfordern.

**Bonus: Speicher beim Erstellen oder Duplizieren des Space anfordern!**

```py
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio"
...     space_storage="large",
... )
```

```py
>>> api.duplicate_space(
...     from_id=repo_id,
...     storage="large",
... )
```

## Fortgeschritten: Temporäres Space Upgrade

Spaces ermöglichen viele verschiedene Einsatzmöglichkeiten. Manchmal möchten Sie vielleicht einen Space vorübergehend auf einer
bestimmten Hardware ausführen, etwas tun und ihn dann herunterfahren. In diesem Abschnitt werden wir untersuchen, wie Sie die
Vorteile von Spaces nutzen können, um ein Modell auf Abruf zu finetunen. Dies ist nur eine Möglichkeit, dieses spezielle Problem zu
lösen. Es sollte als Vorschlag betrachtet und an Ihren Anwendungsfall angepasst werden.


Nehmen wir an, wir haben einen Space, um ein Modell zu finetunen.
Es handelt sich um eine Gradio-App, die ein Modell-Id und eine Dataset-Id als Eingabe nimmt. Der Ablauf sieht folgendermaßen aus:

0. (Den Benutzer nach einem Modell und einem Datensatz auffordern)
1. Das Modell aus dem Hub laden.
2. Den Datensatz aus dem Hub laden.
3. Das Modell mit dem Datensatz finetunen.
4. Das neue Modell auf den Hub hochladen.

Schritt 3 erfordert eine spezielle Hardware, aber Sie möchten nicht, dass Ihr Space die ganze Zeit
auf einer kostenpflichtigen GPU läuft. Eine Lösung besteht darin, dynamisch Hardware für das Training
anzufordern und es anschließend herunterzufahren. Da das Anfordern von Hardware Ihren Space neu startet,
muss sich Ihre App irgendwie die aktuelle Aufgabe "merken", die sie ausführt.
Es gibt mehrere Möglichkeiten, dies zu tun. In diesem Leitfaden sehen wir eine Lösung,
bei der ein Datensatz als "Aufgabenplaner (task scheduler)" verwendet wird.

### App-Grundgerüst

So würde Ihre App aussehen. Beim Start überprüfen, ob eine Aufgabe geplant ist und ob ja,
führen Sie sie auf der richtigen Hardware aus. Ist die Aufgabe erledigt,
setzen Sie die Hardware zurück auf den kostenlosen CPU-Plan und fordern den Benutzer auf,
eine neue Aufgabe anzufordern.

> [!WARNING]
> Ein solcher Workflow unterstützt keinen gleichzeitigen Zugriff wie normale Demos.
> Insbesondere wird die Schnittstelle deaktiviert, wenn das Training stattfindet.
> Es ist vorzuziehen, Ihr Repo auf privat zu setzen, um sicherzustellen, dass Sie der einzige Benutzer sind.

```py
# Für den Space wird Ihr Token benötigt, um Hardware anzufordern: Legen Sie es als Geheimnis fest!
HF_TOKEN = os.environ.get("HF_TOKEN")

# Eigene repo_id des Space
TRAINING_SPACE_ID = "Wauplin/dreambooth-training"

from huggingface_hub import HfApi, SpaceHardware
api = HfApi(token=HF_TOKEN)

# Beim Start des Space überprüfen, ob eine Aufgabe geplant ist. Wenn ja, finetunen Sie das Modell.
# Wenn nicht, zeigen Sie eine Schnittstelle an, um eine neue Aufgabe anzufordern.
task = get_task()
if task is None:
    # Starten der Gradio-App
    def gradio_fn(task):
        # Bei Benutzeranfrage, Aufgabe hinzufügen und Hardware anfordern
        add_task(task)
        api.request_space_hardware(repo_id=TRAINING_SPACE_ID, hardware=SpaceHardware.T4_MEDIUM)

    gr.Interface(fn=gradio_fn, ...).launch()
else:
    runtime = api.get_space_runtime(repo_id=TRAINING_SPACE_ID)
    # Überprüfen, ob der Space mit einer GPU geladen ist.
    if runtime.hardware == SpaceHardware.T4_MEDIUM:
        # Wenn ja, finetunen des Basismodells auf den Datensatz!
        train_and_upload(task)

        # Dann die Aufgabe als "DONE / ERLEDIGT" markieren
        mark_as_done(task)

        # NICHT VERGESSEN: CPU-Hardware zurück setzen
        api.request_space_hardware(repo_id=TRAINING_SPACE_ID, hardware=SpaceHardware.CPU_BASIC)
    else:
        api.request_space_hardware(repo_id=TRAINING_SPACE_ID, hardware=SpaceHardware.T4_MEDIUM)
```

### Aufgabenplaner (Task scheduler)

Das Planen von Aufgaben kann auf viele Arten erfolgen. Hier ist ein Beispiel,
wie es mit einer einfachen CSV gemacht werden könnte, die als Datensatz gespeichert ist.

```py
# Dataset-ID, in der eine `tasks.csv` Datei die auszuführenden Aufgaben enthält.
# Hier ist ein einfaches Beispiel für `tasks.csv`, das Eingaben (Basis-Modell und Datensatz)
# und Status (PENDING / AUSSTEHEND oder DONE / ERLEDIGT) enthält.
#     multimodalart/sd-fine-tunable,Wauplin/concept-1,DONE
#     multimodalart/sd-fine-tunable,Wauplin/concept-2,PENDING
TASK_DATASET_ID = "Wauplin/dreambooth-task-scheduler"

def _get_csv_file():
    return hf_hub_download(repo_id=TASK_DATASET_ID, filename="tasks.csv", repo_type="dataset", token=HF_TOKEN)

def get_task():
    with open(_get_csv_file()) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[2] == "PENDING":
                return row[0], row[1] # model_id, dataset_id

def add_task(task):
    model_id, dataset_id = task
    with open(_get_csv_file()) as csv_file:
        with open(csv_file, "r") as f:
            tasks = f.read()

    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo="tasks.csv",
        # Schnelle und einfache Möglichkeit, eine Aufgabe hinzuzufügen
        path_or_fileobj=(tasks + f"\n{model_id},{dataset_id},PENDING").encode()
    )

def mark_as_done(task):
    model_id, dataset_id = task
    with open(_get_csv_file()) as csv_file:
        with open(csv_file, "r") as f:
            tasks = f.read()

    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo="tasks.csv",
        # Schnelle und einfache Möglichkeit, die Aufgabe als DONE / ERLEDIGT zu markieren
        path_or_fileobj=tasks.replace(
            f"{model_id},{dataset_id},PENDING",
            f"{model_id},{dataset_id},DONE"
        ).encode()
    )
```
