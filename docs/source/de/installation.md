<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Installation

Bevor Sie beginnen, müssen Sie Ihre Umgebung vorbereiten, indem Sie die entsprechenden Pakete installieren.

`huggingface_hub` wurde für **Python 3.9+** getestet.

## Installation mit pip

Es wird dringend empfohlen, `huggingface_hub` in einer [virtuellen Umgebung](https://docs.python.org/3/library/venv.html) zu installieren. Wenn Sie mit virtuellen Umgebungen in Python nicht vertraut sind, werfen Sie einen Blick auf diesen [Leitfaden](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/). Eine virtuelle Umgebung erleichtert die Verwaltung verschiedener Projekte und verhindert Kompatibilitätsprobleme zwischen Abhängigkeiten.

Beginnen Sie damit, eine virtuelle Umgebung in Ihrem Projektverzeichnis zu erstellen:

```bash
python -m venv .env
```

Aktivieren Sie die virtuelle Umgebung. Unter Linux und macOS:

```bash
source .env/bin/activate
```

Aktivieren der virtuellen Umgebung unter Windows:

```bash
.env/Scripts/activate
```

Jetzt können Sie `huggingface_hub` aus dem [PyPi-Register](https://pypi.org/project/huggingface-hub/) installieren:

```bash
pip install --upgrade huggingface_hub
```

Überprüfen Sie nach Abschluss, ob die [Installation korrekt funktioniert](#installation-berprfen).

### Installieren optionaler Abhängigkeiten

Einige Abhängigkeiten von `huggingface_hub` sind [optional](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies), da sie nicht notwendig sind, um die Kernfunktionen von `huggingface_hub` auszuführen. Allerdings könnten einige Funktionen von `huggingface_hub` nicht verfügbar sein, wenn die optionalen Abhängigkeiten nicht installiert sind.

Sie können optionale Abhängigkeiten über `pip` installieren:
```bash
# Abhängigkeiten sowohl für torch-spezifische als auch für CLI-spezifische Funktionen installieren.
pip install 'huggingface_hub[cli,torch]'
```

Hier ist die Liste der optionalen Abhängigkeiten in huggingface_hub:

- `cli`: bietet eine komfortablere CLI-Schnittstelle für huggingface_hub.
- `fastai`, `torch`: Abhängigkeiten, um framework-spezifische Funktionen auszuführen.
- `dev`: Abhängigkeiten, um zur Bibliothek beizutragen. Enthält `testing` (um Tests auszuführen), `typing` (um den Type Checker auszuführen) und `quality` (um Linters auszuführen).


### Installieren von der Quelle

In einigen Fällen kann es sinnvoll sein, `huggingface_hub` direkt von der Quelle zu installieren. Dies ermöglicht es Ihnen, die aktuellste `main`-Version anstelle der neuesten stabilen Version zu verwenden. Die `main`-Version ist nützlich, um immer auf dem neuesten Stand der Entwicklungen zu bleiben, zum Beispiel wenn ein Fehler seit der letzten offiziellen Veröffentlichung behoben wurde, aber noch keine neue Version herausgegeben wurde.

Das bedeutet jedoch, dass die `main`-Version nicht immer stabil sein könnte. Wir bemühen uns, die Hauptversion funktionsfähig zu halten, und die meisten Probleme werden in der Regel innerhalb von einigen Stunden oder einem Tag gelöst. Wenn Sie auf ein Problem stoßen, eröffnen Sie bitte ein "Issue", damit wir es noch schneller beheben können!

```bash
pip install git+https://github.com/huggingface/huggingface_hub
```

Bei der Installation von der Quelle können Sie auch einen bestimmten Zweig angeben. Dies ist nützlich, wenn Sie ein neues Feature oder einen neuen Fehlerbehebung testen möchten, der noch nicht zusammengeführt wurde:

```bash
pip install git+https://github.com/huggingface/huggingface_hub@my-feature-branch
```

Überprüfen Sie nach Abschluss, ob die [Installation korrekt funktioniert](#installation-berprfen).

### Editierbare Installation

Die Installation von der Quelle ermöglicht Ihnen eine [editierbare Installation](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs). Dies ist eine fortgeschrittenere Installation, wenn Sie zur Entwicklung von `huggingface_hub` beitragen und Änderungen im Code testen möchten. Sie müssen eine lokale Kopie von `huggingface_hub` auf Ihrem Computer klonen.

```bash
# Zuerst die Repository lokal klonen
git clone https://github.com/huggingface/huggingface_hub.git

# Dann mit dem -e Flag installieren
cd huggingface_hub
pip install -e .
```

Diese Befehle verknüpfen den Ordner, in den Sie das Repository geklont haben, mit Ihren Python-Bibliothekspfaden. Python wird nun zusätzlich zu den normalen Bibliothekspfaden im geklonten Ordner suchen. Wenn Ihre Python-Pakete normalerweise in `./.venv/lib/python3.13/site-packages/` installiert sind, wird Python auch den geklonten Ordner `./huggingface_hub/` durchsuchen.

## Installieren mit conda

Wenn Sie damit vertrauter sind, können Sie `huggingface_hub` über den [conda-forge-Kanal](https://anaconda.org/conda-forge/huggingface_hub) installieren:

```bash
conda install -c conda-forge huggingface_hub
```

Überprüfen Sie nach Abschluss, ob die [Installation korrekt funktioniert](#installation-berprfen).

## Installation überprüfen

Nach der Installation überprüfen Sie, ob `huggingface_hub` richtig funktioniert, indem Sie den folgenden Befehl ausführen:

```bash
python -c "from huggingface_hub import model_info; print(model_info('gpt2'))"
```

Dieser Befehl ruft Informationen vom Hub über das [gpt2](https://huggingface.co/gpt2)-Modell ab. Die Ausgabe sollte so aussehen:

```text
Model Name: gpt2
Tags: ['pytorch', 'tf', 'jax', 'tflite', 'rust', 'safetensors', 'gpt2', 'text-generation', 'en', 'doi:10.57967/hf/0039', 'transformers', 'exbert', 'license:mit', 'has_space']
Task: text-generation
```

## Windows-Einschränkungen

Mit unserem Ziel, gutes ML überall zu demokratisieren, haben wir `huggingface_hub` als plattformübergreifende Bibliothek entwickelt, insbesondere um sowohl auf Unix-basierten als auch auf Windows-Systemen korrekt zu funktionieren. Es gibt jedoch einige Fälle, in denen `huggingface_hub` unter Windows gewisse Einschränkungen hat. Hier ist eine ausführliche Liste der bekannten Probleme. Bitte informieren Sie uns, wenn Sie auf ein nicht dokumentiertes Problem stoßen, indem Sie ein [Issue auf Github eröffnen](https://github.com/huggingface/huggingface_hub/issues/new/choose).


- Das Cache-System von `huggingface_hub` verwendet Symlinks, um Dateien, die vom Hub heruntergeladen wurden, effizient zu cachen. Unter Windows müssen Sie den Entwicklermodus aktivieren oder Ihr Skript als Admin ausführen, um Symlinks zu aktivieren. Wenn sie nicht aktiviert sind, funktioniert das Cache-System immer noch, aber nicht optimiert. Bitte lesen Sie den Abschnitt über [Cache-Einschränkungen](./guides/manage-cache#limitations) für weitere Details.
- Dateipfade auf dem Hub können Sonderzeichen enthalten (z.B. `"pfad/zu?/meiner/datei"`). Windows ist bei [Sonderzeichen](https://learn.microsoft.com/en-us/windows/win32/intl/character-sets-used-in-file-names) restriktiver, wodurch es unmöglich ist, diese Dateien unter Windows herunterzuladen. Hoffentlich ist dies ein seltener Fall. Bitte wenden Sie sich an den Repo-Eigentümer, wenn Sie denken, dass dies ein Fehler ist, oder an uns, um eine Lösung zu finden.


## Nächste Schritte

Sobald `huggingface_hub`` richtig auf Ihrem Computer installiert ist, möchten Sie vielleicht [Umgebungsvariablen konfigurieren](package_reference/environment_variables) oder [einen unserer Leitfäden durchgehen](guides/overview), um loszulegen.
