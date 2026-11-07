<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub.svg">
    <img alt="huggingface_hub library logo" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p> 

<p align="center">
    <i>Der offizielle Python-Client für den Huggingface Hub.</i>
</p>

<p align="center">
    <a href="https://huggingface.co/docs/huggingface_hub/de/index"><img alt="Dokumentation" src="https://img.shields.io/website/http/huggingface.co/docs/huggingface_hub/index.svg?down_color=red&down_message=offline&up_message=online&label=doc"></a>
    <a href="https://github.com/huggingface/huggingface_hub/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/huggingface_hub.svg"></a>
    <a href="https://github.com/huggingface/huggingface_hub"><img alt="PyPi version" src="https://img.shields.io/pypi/pyversions/huggingface_hub.svg"></a>
    <a href="https://pypi.org/project/huggingface-hub"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/huggingface_hub"></a>
    <a href="https://codecov.io/gh/huggingface/huggingface_hub"><img alt="Code coverage" src="https://codecov.io/gh/huggingface/huggingface_hub/branch/main/graph/badge.svg?token=RXP95LE2XL"></a>
</p>

<h4 align="center">
    <p>
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/README.md">English</a>  |
        <b>Deutsch</b> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_hi.md">हिंदी</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_ko.md">한국인</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_cn.md">中文（简体）</a>
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_kn.md">ಕನ್ನಡ</a>
    <p>
</h4>
---

**Dokumentation**: <a href="https://hf.co/docs/huggingface_hub" target="_blank">https://hf.co/docs/huggingface_hub</a>

**Quellcode**: <a href="https://github.com/huggingface/huggingface_hub" target="_blank">https://github.com/huggingface/huggingface_hub</a>

---

## Willkommen bei der huggingface_hub Bibliothek

Die `huggingface_hub` Bibliothek ermöglicht Ihnen die Interaktion mit dem [Hugging Face Hub](https://huggingface.co/), einer Plattform, die Open-Source Machine Learning für Entwickler und Mitwirkende demokratisiert. Entdecken Sie vortrainierte Modelle und Datensätze für Ihre Projekte oder spielen Sie mit den Tausenden von Machine-Learning-Apps, die auf dem Hub gehostet werden. Sie können auch Ihre eigenen Modelle, Datensätze und Demos mit der Community teilen. Die `huggingface_hub` Bibliothek bietet eine einfache Möglichkeit, all dies mit Python zu tun.

## Hauptmerkmale

- Dateien vom Hub [herunterladen](https://huggingface.co/docs/huggingface_hub/de/guides/download).
- Dateien auf den Hub [hochladen](https://huggingface.co/docs/huggingface_hub/de/guides/upload).
- [Verwalten Ihrer Repositories](https://huggingface.co/docs/huggingface_hub/de/guides/repository).
- [Ausführen von Inferenz](https://huggingface.co/docs/huggingface_hub/de/guides/inference) auf bereitgestellten Modellen.
- [Suche](https://huggingface.co/docs/huggingface_hub/de/guides/search) nach Modellen, Datensätzen und Spaces.
- [Model Cards teilen](https://huggingface.co/docs/huggingface_hub/de/guides/model-cards), um Ihre Modelle zu dokumentieren.
- [Mit der Community interagieren](https://huggingface.co/docs/huggingface_hub/de/guides/community), durch PRs und Kommentare.

## Installation

Installieren Sie das `huggingface_hub` Paket mit [pip](https://pypi.org/project/huggingface-hub/):

```bash
pip install huggingface_hub
```

Wenn Sie möchten, können Sie es auch mit [conda](https://huggingface.co/docs/huggingface_hub/de/installation#installieren-mit-conda) installieren.

Um das Paket standardmäßig minimal zu halten, kommt `huggingface_hub` mit optionalen Abhängigkeiten, die für einige Anwendungsfälle nützlich sind. Zum Beispiel, wenn Sie ein vollständiges Erlebnis für Inferenz möchten, führen Sie den folgenden Befehl aus:

```bash
pip install huggingface_hub[inference]
```

Um mehr über die Installation und optionale Abhängigkeiten zu erfahren, sehen Sie sich bitte den [Installationsleitfaden](https://huggingface.co/docs/huggingface_hub/de/installation) an.

## Schnellstart

### Dateien herunterladen

Eine einzelne Datei herunterladen

```py
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="tiiuae/falcon-7b-instruct", filename="config.json")
```

Oder eine gesamte Repository

```py
from huggingface_hub import snapshot_download

snapshot_download("stabilityai/stable-diffusion-2-1")
```

Dateien werden in einen lokalen Cache-Ordner heruntergeladen. Weitere Details finden Sie in diesem [Leitfaden](https://huggingface.co/docs/huggingface_hub/de/guides/manage-cache).

### Anmeldung

Der Hugging Face Hub verwendet Tokens zur Authentifizierung von Anwendungen (siehe [Dokumentation](https://huggingface.co/docs/hub/security-tokens)). Um sich an Ihrem Computer anzumelden, führen Sie das folgende Kommando in der Befehlszeile aus:

```bash
hf auth login
# oder mit einer Umgebungsvariablen
hf auth login --token $HUGGINGFACE_TOKEN
```

### Eine Repository erstellen

```py
from huggingface_hub import create_repo

create_repo(repo_id="super-cool-model")
```

### Dateien hochladen

Eine einzelne Datei hochladen

```py
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="/home/lysandre/dummy-test/README.md",
    path_in_repo="README.md",
    repo_id="lysandre/test-model",
)
```

Oder einen gesamten Ordner

```py
from huggingface_hub import upload_folder

upload_folder(
    folder_path="/path/to/local/space",
    repo_id="username/my-cool-space",
    repo_type="space",
)
```

Weitere Informationen finden Sie im [Upload-Leitfaden](https://huggingface.co/docs/huggingface_hub/de/guides/upload).

## Integration in den Hub

Wir arbeiten mit coolen Open-Source-ML-Bibliotheken zusammen, um kostenloses Model-Hosting und -Versionierung anzubieten. Die bestehenden Integrationen finden Sie [hier](https://huggingface.co/docs/hub/libraries).

Die Vorteile sind:

- Kostenloses Hosting von Modellen oder Datensätzen für Bibliotheken und deren Benutzer..
- Eingebaute Dateiversionierung, selbst bei sehr großen Dateien, dank eines git-basierten Ansatzes.
- Bereitgestellte Inferenz-API für alle öffentlich verfügbaren Modelle.
- In-Browser-Widgets zum Spielen mit den hochgeladenen Modellen.
- Jeder kann ein neues Modell für Ihre Bibliothek hochladen, es muss nur das entsprechende Tag hinzugefügt werden, damit das Modell auffindbar ist.
- Schnelle Downloads! Wir verwenden Cloudfront (ein CDN), um Downloads zu geo-replizieren, sodass sie von überall auf der Welt blitzschnell sind.
- Nutzungsstatistiken und mehr Funktionen in Kürze.

Wenn Sie Ihre Bibliothek integrieren möchten, öffnen Sie gerne ein Issue, um die Diskussion zu beginnen. Wir haben mit ❤️ einen [schrittweisen Leitfaden](https://huggingface.co/docs/hub/adding-a-library) geschrieben, der zeigt, wie diese Integration durchgeführt wird.

## Beiträge (Feature-Anfragen, Fehler usw.) sind super willkommen 💙💚💛💜🧡❤️

Jeder ist willkommen beizutragen, und wir schätzen den Beitrag jedes Einzelnen. Code zu schreiben ist nicht der einzige Weg, der Community zu helfen. Fragen zu beantworten, anderen zu helfen, sich zu vernetzen und die Dokumentationen zu verbessern, sind für die Gemeinschaft von unschätzbarem Wert. Wir haben einen [Beitrags-Leitfaden](https://github.com/huggingface/huggingface_hub/blob/main/CONTRIBUTING.md) geschrieben, der zusammenfasst, wie Sie beginnen können, zu dieser Repository beizutragen.
