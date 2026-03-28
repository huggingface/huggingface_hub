<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Integrieren Sie jedes ML-Framework mit dem Hub

Der Hugging Face Hub erleichtert das Hosten und Teilen von Modellen mit der Community. Er unterstützt [Dutzende von Bibliotheken](https://huggingface.co/docs/hub/models-libraries) im Open Source-Ökosystem. Wir arbeiten ständig daran, diese Unterstützung zu erweitern, um kollaboratives Machine Learning voranzutreiben. Die `huggingface_hub`-Bibliothek spielt eine Schlüsselrolle in diesem Prozess und ermöglicht es jedem Python-Skript, Dateien einfach hochzuladen und zu laden.

Es gibt vier Hauptwege, eine Bibliothek mit dem Hub zu integrieren:
1. **Push to Hub**: Implementieren Sie eine Methode, um ein Modell auf den Hub hochzuladen.
   Dies beinhaltet das Modellgewicht sowie [die Modellkarte](https://huggingface.co/docs/huggingface_hub/how-to-model-cards) und alle anderen relevanten Informationen oder Daten, die für den Betrieb des Modells erforderlich sind (zum Beispiel Trainingsprotokolle). Diese Methode wird oft `push_to_hub()` genannt.
2. **Download from Hub**: Implementieren Sie eine Methode, um ein Modell vom Hub zu laden.
   Die Methode sollte die Modellkonfiguration/-gewichte herunterladen und das Modell laden. Diese Methode wird oft `from_pretrained` oder `load_from_hub()` genannt.
3. **Widgets**: Zeigen Sie ein Widget auf der Landing Page Ihrer Modelle auf dem Hub an.
   Dies ermöglicht es Benutzern, ein Modell schnell aus dem Browser heraus auszuprobieren.

In diesem Leitfaden konzentrieren wir uns auf die ersten beiden Themen. Wir werden die beiden Hauptansätze vorstellen, die Sie zur Integration einer Bibliothek verwenden können, mit ihren Vor- und Nachteilen. Am Ende des Leitfadens ist alles zusammengefasst, um Ihnen bei der Auswahl zwischen den beiden zu helfen. Bitte beachten Sie, dass dies nur Richtlinien sind, die Sie an Ihre Anforderungen anpassen können.

Wenn Sie sich für Inferenz und Widgets interessieren, können Sie [diesem Leitfaden](https://huggingface.co/docs/hub/models-adding-libraries#set-up-the-inference-api) folgen. In beiden Fällen können Sie sich an uns wenden, wenn Sie eine Bibliothek mit dem Hub integrieren und [in unserer Dokumentation](https://huggingface.co/docs/hub/models-libraries) aufgeführt haben möchten.

## Ein flexibler Ansatz: Helfer

Der erste Ansatz zur Integration einer Bibliothek in den Hub besteht tatsächlich darin, die `push_to_hub` und `from_pretrained` Methoden selbst zu implementieren. Dies gibt Ihnen volle Flexibilität hinsichtlich der Dateien, die Sie hoch-/herunterladen möchten, und wie Sie Eingaben, die speziell für Ihr Framework sind, behandeln. Sie können sich die beiden Leitfäden [Dateien hochladen](./upload) und [Dateien herunterladen](./download) ansehen, um mehr darüber zu erfahren, wie dies funktioniert. Dies ist zum Beispiel die Art und Weise, wie die FastAI-Integration implementiert ist (siehe [`push_to_hub_fastai`]
und [`from_pretrained_fastai`]).

Die Implementierung kann zwischen den Bibliotheken variieren, aber der Workflow ist oft ähnlich.

### from_pretrained

So sieht eine `from_pretrained` Methode normalerweise aus:

```python
def from_pretrained(model_id: str) -> MyModelClass:
   # Modell vom Hub herunterladen
   cached_model = hf_hub_download(
      repo_id=repo_id,
      filename="model.pkl",
      library_name="fastai",
      library_version=get_fastai_version(),
   )

   # Modell laden
    return load_model(cached_model)
```

### push_to_hub

Die `push_to_hub` Methode erfordert oft etwas mehr Komplexität, um die Repo-Erstellung, die Generierung der Modellkarte und das Speichern von Gewichten zu behandeln. Ein üblicher Ansatz besteht darin, all diese Dateien in einem temporären Ordner zu speichern, ihn hochzuladen und dann zu löschen.

```python
def push_to_hub(model: MyModelClass, repo_name: str) -> None:
   api = HfApi()

   # Repo erstellen, wenn noch nicht vorhanden und die zugehörige repo_id erhalten
   repo_id = api.create_repo(repo_name, exist_ok=True)

   # Modell in temporärem Ordner speichern und in einem enzigen Commit pushen
   with TemporaryDirectory() as tmpdir:
      tmpdir = Path(tmpdir)

      # Gewichte speichern
      save_model(model, tmpdir / "model.safetensors")

      # Modellkarte generieren
      card = generate_model_card(model)
      (tmpdir / "README.md").write_text(card)

      # Logs speichern
      # Diagramme speichern
      # Evaluationsmetriken speichern
      # ...

      # Auf den Hub pushen
      return api.upload_folder(repo_id=repo_id, folder_path=tmpdir)
```

Dies ist natürlich nur ein Beispiel. Wenn Sie an komplexeren Manipulationen interessiert sind (entfernen von entfernten Dateien, hochladen von Gewichten on-the-fly, lokales Speichern von Gewichten, usw.), beachten Sie bitte den [Dateien hochladen](./upload) Leitfaden.

### Einschränkungen

Obwohl dieser Ansatz flexibel ist, hat er einige Nachteile, insbesondere in Bezug auf die Wartung. Hugging Face-Benutzer sind oft an zusätzliche Funktionen gewöhnt, wenn sie mit `huggingface_hub` arbeiten. Zum Beispiel ist es beim Laden von Dateien aus dem Hub üblich, Parameter wie folgt anzubieten:
- `token`: zum Herunterladen aus einem privaten Repository
- `revision`: zum Herunterladen von einem spezifischen Branch
- `cache_dir`: um Dateien in einem spezifischen Verzeichnis zu cachen
- `force_download`/`local_files_only`: um den Cache wieder zu verwenden oder nicht
- `api_endpoint`/`proxies`: HTTP-Session konfigurieren

Beim Pushen von Modellen werden ähnliche Parameter unterstützt:
- `commit_message`: benutzerdefinierte Commit-Nachricht
- `private`: ein privates Repository erstellen, falls nicht vorhanden
- `create_pr`: erstellen Sie einen PR anstatt auf `main` zu pushen
- `branch`: auf einen Branch pushen anstatt auf den `main` Branch
- `allow_patterns`/`ignore_patterns`: filtern, welche Dateien hochgeladen werden sollen
- `token`
- `api_endpoint`
- ...

Alle diese Parameter können den zuvor gesehenen Implementierungen hinzugefügt und an die `huggingface_hub`-Methoden übergeben werden.
 Wenn sich jedoch ein Parameter ändert oder eine neue Funktion hinzugefügt wird, müssen Sie Ihr Paket aktualisieren.
 Die Unterstützung dieser Parameter bedeutet auch mehr Dokumentation, die Sie auf Ihrer Seite pflegen müssen.
 Um zu sehen, wie man diese Einschränkungen mildert, springen wir zu unserem nächsten Abschnitt **Klassenvererbung**.

## Ein komplexerer Ansatz: Klassenvererbung

Wie wir oben gesehen haben, gibt es zwei Hauptmethoden, um Ihre Bibliothek mit dem Hub zu integrieren: Dateien hochladen (`push_to_hub`) und Dateien herunterladen (`from_pretrained`). Sie können diese Methoden selbst implementieren, aber das hat seine Tücken. Um dies zu bewältigen, bietet `huggingface_hub` ein Werkzeug an, das Klassenvererbung verwendet. Schauen wir uns an, wie es funktioniert!

In vielen Fällen implementiert eine Bibliothek ihr Modell bereits mit einer Python-Klasse. Die Klasse enthält die Eigenschaften des Modells und Methoden zum Laden, Ausführen, Trainieren und Evaluieren. Unser Ansatz besteht darin, diese Klasse zu erweitern, um Upload- und Download-Funktionen mit Mixins hinzuzufügen. Ein [Mixin](https://stackoverflow.com/a/547714) ist eine Klasse, die dazu bestimmt ist, eine vorhandene Klasse mit einem Satz spezifischer Funktionen durch Mehrfachvererbung zu erweitern. `huggingface_hub` bietet sein eigenes Mixin, das [`ModelHubMixin`]. Der Schlüssel hier ist zu verstehen, wie es funktioniert und wie man es anpassen kann.

Die Klasse [ModelHubMixin] implementiert 3 *öffentliche* Methoden (`push_to_hub`, `save_pretrained` und `from_pretrained`). Dies sind die Methoden, die Ihre Benutzer aufrufen werden, um Modelle mit Ihrer Bibliothek zu laden/speichern. [`ModelHubMixin`] definiert auch 2 private Methoden (`_save_pretrained` und `_from_pretrained`). Diese müssen Sie implementieren. Um Ihre Bibliothek zu integrieren, sollten Sie:

1. Lassen Sie Ihre Modell-Klasse von [`ModelHubMixin`] erben.
2. Implementieren Sie die privaten Methoden:
   - [`~ModelHubMixin._save_pretrained`]: Methode, die als Eingabe einen Pfad zu einem Verzeichnis nimmt und das Modell dort speichert. Sie müssen die gesamte Logik zum Speichern Ihres Modells in dieser Methode schreiben: Modellkarte, Modellgewichte, Konfigurationsdateien, Trainingsprotokolle und Diagramme. Alle relevanten Informationen für dieses Modell müssen von dieser Methode behandelt werden.
   [Model Cards](https://huggingface.co/docs/hub/model-cards) sind besonders wichtig, um Ihr Modell zu beschreiben. Weitere Details finden Sie in [unserem Implementierungsleitfaden](./model-cards).
   - [~ModelHubMixin._from_pretrained]: **Klassenmethode**, die als Eingabe eine `model_id` nimmt und ein instanziiertes Modell zurückgibt. Die Methode muss die relevanten Dateien herunterladen und laden.
3. Sie sind fertig!

Der Vorteil der Verwendung von [`ModelHubMixin`] besteht darin, dass Sie, sobald Sie sich um die Serialisierung/das Laden der Dateien gekümmert haben, bereit sind los zu legen. Sie müssen sich keine Gedanken über Dinge wie Repository-Erstellung, Commits, PRs oder Revisionen machen. All dies wird von dem Mixin gehandhabt und steht Ihren Benutzern zur Verfügung. Das Mixin stellt auch sicher, dass öffentliche Methoden gut dokumentiert und typisiert sind.

### Ein konkretes Beispiel: PyTorch

Ein gutes Beispiel für das, was wir oben gesehen haben, ist [`PyTorchModelHubMixin`], unsere Integration für das PyTorch-Framework. Dies ist eine einsatzbereite Integration.

#### Wie verwendet man es?

Hier ist, wie jeder Benutzer ein PyTorch-Modell vom/auf den Hub laden/speichern kann:

```python
>>> import torch
>>> import torch.nn as nn
>>> from huggingface_hub import PyTorchModelHubMixin

# 1. Definieren Sie Ihr Pytorch-Modell genau so, wie Sie es gewohnt sind
>>> class MyModel(nn.Module, PyTorchModelHubMixin): # Mehrfachvererbung
...     def __init__(self):
...         super().__init__()
...         self.param = nn.Parameter(torch.rand(3, 4))
...         self.linear = nn.Linear(4, 5)

...     def forward(self, x):
...         return self.linear(x + self.param)
>>> model = MyModel()

# 2. (optional) Modell in lokales Verzeichnis speichern
>>> model.save_pretrained("path/to/my-awesome-model")

# 3. Modellgewichte an den Hub übertragen
>>> model.push_to_hub("my-awesome-model")

# 4. Modell vom Hub initialisieren
>>> model = MyModel.from_pretrained("username/my-awesome-model")
```

#### Implementierung

Die Implementierung ist tatsächlich sehr einfach, und die vollständige Implementierung finden Sie [hier](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hub_mixin.py).

1. Zuerst, erben Ihrer Klasse von `ModelHubMixin`:

```python
from huggingface_hub import ModelHubMixin

class PyTorchModelHubMixin(ModelHubMixin):
   (...)
```

2. Implementieren der `_save_pretrained` Methode:

```py
from huggingface_hub import ModelCard, ModelCardData

class PyTorchModelHubMixin(ModelHubMixin):
   (...)

   def _save_pretrained(self, save_directory: Path):
      """Generiere Modellkarte und speichere Gewichte von einem Pytorch-Modell in einem lokalen Verzeichnis."""
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

3. Implementieren der `_from_pretrained` Methode:

```python
class PyTorchModelHubMixin(ModelHubMixin):
   (...)

   @classmethod # Muss eine Klassenmethode sein!
   def _from_pretrained(
      cls,
      *,
      model_id: str,
      revision: str,
      cache_dir: str,
      force_download: bool,
      proxies: Optional[dict],
      local_files_only: bool,
      token: Union[str, bool, None],
      map_location: str = "cpu", # zusätzliches Argument
      strict: bool = False, # zusätzliches Argument
      **model_kwargs,
   ):
      """Load Pytorch pretrained weights and return the loaded model."""
      if os.path.isdir(model_id): # Kann entweder ein lokales Verzeichnis sein
         print("Loading weights from local directory")
         model_file = os.path.join(model_id, "pytorch_model.bin")
      else: # Oder ein Modell am Hub
         model_file = hf_hub_download( # Herunterladen vom Hub, gleiche Eingabeargumente
            repo_id=model_id,
            filename="pytorch_model.bin",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
         )

      # Modell laden und zurückgeben - benutzerdefinierte Logik je nach Ihrem Framework
      model = cls(**model_kwargs)
      state_dict = torch.load(model_file, map_location=torch.device(map_location))
      model.load_state_dict(state_dict, strict=strict)
      model.eval()
      return model
```

Und das war's! Ihre Bibliothek ermöglicht es Benutzern nun, Dateien vom und zum Hub hoch- und herunterzuladen.

## Kurzer Vergleich

Lassen Sie uns die beiden Ansätze, die wir gesehen haben, schnell mit ihren Vor- und Nachteilen zusammenfassen. Die untenstehende Tabelle ist nur indikativ. Ihr Framework könnte einige Besonderheiten haben, die Sie berücksichtigen müssen. Dieser Leitfaden soll nur Richtlinien und Ideen geben, wie Sie die Integration handhaben können. Kontaktieren Sie uns in jedem Fall, wenn Sie Fragen haben!

<!-- Generated using https://www.tablesgenerator.com/markdown_tables -->
|         Integration          |                                                                 Mit Helfern                                                                 |                                       Mit [`ModelHubMixin`]                                        |
| :--------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------: |
|      Benutzererfahrung       |                                          `model = load_from_hub(...)`<br>`push_to_hub(model, ...)`                                          |                 `model = MyModel.from_pretrained(...)`<br>`model.push_to_hub(...)`                 |
|         Flexibilität         |                                  Sehr flexibel.<br>Sie haben die volle Kontrolle über die Implementierung.                                  |                  Weniger flexibel.<br>Ihr Framework muss eine Modellklasse haben.                  |
|           Wartung            | Mehr Wartung, um Unterstützung für Konfiguration und neue Funktionen hinzuzufügen. Könnte auch das Beheben von Benutzerproblemen erfordern. | Weniger Wartung, da die meisten Interaktionen mit dem Hub in `huggingface_hub` implementiert sind. |
| Dokumentation/Typ-Annotation |                                                            Manuell zu schreiben.                                                            |                            Teilweise durch `huggingface_hub` behandelt.                            |
