<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Kurzanleitung

Der [Hugging Face Hub](https://huggingface.co/) ist die erste Anlaufstelle für das Teilen von Maschinenlernmodellen, Demos, Datensätzen und Metriken. Die `huggingface_hub`-Bibliothek hilft Ihnen, mit dem Hub zu interagieren, ohne Ihre Entwicklungs-Umgebung zu verlassen. Sie können Repositories einfach erstellen und verwalten, Dateien herunterladen und hochladen und nützliche Model- und Datensatz-Metadaten vom Hub abrufen.

## Installation

Um loszulegen, installieren Sie die `huggingface_hub`-Bibliothek:

```bash
pip install --upgrade huggingface_hub
```

Für weitere Details schauen Sie sich bitte den [Installationsleitfaden](installation) an.

## Dateien herunterladen

Repositories auf dem Hub sind mit git versioniert, und Benutzer können eine einzelne Datei
oder das gesamte Repository herunterladen. Sie können die Funktion [`hf_hub_download`] verwenden, um Dateien herunterzuladen.
Diese Funktion lädt eine Datei herunter und speichert sie im Cache auf Ihrer lokalen Festplatte. Das nächste Mal, wenn Sie diese Datei benötigen, wird sie aus Ihrem Cache geladen, sodass Sie sie nicht erneut herunterladen müssen.

Sie benötigen die Repository-ID und den Dateinamen der Datei, die Sie herunterladen möchten. Zum
Beispiel, um die Konfigurationsdatei des [Pegasus](https://huggingface.co/google/pegasus-xsum) Modells herunterzuladen:

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(repo_id="google/pegasus-xsum", filename="config.json")
```

Um eine bestimmte Version der Datei herunterzuladen, verwenden Sie den `revision`-Parameter, um den
Namen der Branch, des Tags oder des Commit-Hashes anzugeben. Wenn Sie sich für den Commit-Hash
entscheiden, muss es der vollständige Hash anstelle des kürzeren 7-Zeichen-Commit-Hashes sein:

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(
...     repo_id="google/pegasus-xsum",
...     filename="config.json",
...     revision="4d33b01d79672f27f001f6abade33f22d993b151"
... )
```

Für weitere Details und Optionen siehe die API-Referenz für [`hf_hub_download`].

## Anmeldung

In vielen Fällen müssen Sie mit einem Hugging Face-Konto angemeldet sein, um mit dem Hub zu interagieren: private Repos herunterladen, Dateien hochladen, PRs erstellen,...
[Erstellen Sie ein Konto](https://huggingface.co/join), wenn Sie noch keines haben, und melden Sie sich dann an, um Ihr ["User Access Token"](https://huggingface.co/docs/hub/security-tokens) von Ihrer [Einstellungsseite](https://huggingface.co/settings/tokens) zu erhalten. Das "User Access Token" wird verwendet, um Ihre Identität gegenüber dem Hub zu authentifizieren.

Sobald Sie Ihr "User Access Token" haben, führen Sie den folgenden Befehl in Ihrem Terminal aus:

```bash
hf auth login
# or using an environment variable
hf auth login --token $HUGGINGFACE_TOKEN
```

Alternativ können Sie sich auch programmatisch in einem Notebook oder einem Skript mit [`login`] anmelden:

```py
>>> from huggingface_hub import login
>>> login()
```

Es ist auch möglich, sich programmatisch anzumelden, ohne aufgefordert zu werden, Ihr Token einzugeben, indem Sie das Token direkt an [`login`] weitergeben, wie z.B. `login(token="hf_xxx")`. Seien Sie vorsichtig, wenn Sie Ihren Quellcode teilen. Es ist eine bewährte Methode, das Token aus einem sicheren Tresor/Vault zu laden, anstatt es explizit in Ihrer Codebasis/Notebook zu speichern.

Sie können nur auf 1 Konto gleichzeitig angemeldet sein. Wenn Sie Ihren Computer mit einem neuen Konto anmelden, werden Sie vom vorherigen abgemeldet. Mit dem Befehl `hf auth whoami` stellen Sie sicher, dass Sie immer wissen, welches Konto Sie gerade verwenden. Wenn Sie mehrere Konten im selben Skript verwalten möchten, können Sie Ihr Token bereitstellen, wenn Sie jede Methode aufrufen. Dies ist auch nützlich, wenn Sie kein Token auf Ihrem Computer speichern möchten.

> [!WARNING]
> Sobald Sie angemeldet sind, werden alle Anfragen an den Hub - auch Methoden, die nicht unbedingt eine Authentifizierung erfordern - standardmäßig Ihr Zugriffstoken verwenden. Wenn Sie die implizite Verwendung Ihres Tokens deaktivieren möchten, sollten Sie die Umgebungsvariable `HF_HUB_DISABLE_IMPLICIT_TOKEN` setzen.

## Eine Repository erstellen

Nachdem Sie sich registriert und angemeldet haben, können Sie mit der Funktion [`create_repo`] ein Repository erstellen:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model")
```

If you want your repository to be private, then:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model", private=True)
```

Private Repositories sind für niemanden außer Ihnen selbst sichtbar.

> [!TIP]
> Um eine Repository zu erstellen oder Inhalte auf den Hub zu pushen, müssen Sie ein "User Access Token" bereitstellen, das die Schreibberechtigung (`write`) hat. Sie können die Berechtigung auswählen, wenn Sie das Token auf Ihrer [Einstellungsseite](https://huggingface.co/settings/tokens) erstellen.

## Dateien hochladen

Verwenden Sie die [`upload_file`]-Funktion, um eine Datei zu Ihrem neu erstellten Repository hinzuzufügen. Sie müssen dabei das Folgende angeben:

1. Den Pfad der hochzuladenden Datei.
2. Den Pfad der Datei im Repository.
3. Die Repository-ID, zu der Sie die Datei hinzufügen möchten.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.upload_file(
...     path_or_fileobj="/home/lysandre/dummy-test/README.md",
...     path_in_repo="README.md",
...     repo_id="lysandre/test-model",
... )
```

Um mehr als eine Datei gleichzeitig hochzuladen, werfen Sie bitte einen Blick auf den [Upload](./guides/upload)-Leitfaden, der Ihnen verschiedene Methoden zum Hochladen von Dateien vorstellt (mit oder ohne git).

## Nächste Schritte

Die `huggingface_hub`-Bibliothek bietet den Benutzern eine einfache Möglichkeit, mittels Python mit dem Hub zu interagieren. Um mehr darüber zu erfahren, wie Sie Ihre Dateien und Repositories auf dem Hub verwalten können, empfehlen wir, unsere [How-to-Leitfäden](./guides/overview) zu lesen:

- [Verwalten Sie Ihre Repository](./guides/repository).
- Dateien vom Hub [herunterladen](./guides/download).
- Dateien auf den Hub [hochladen](./guides/upload).
- [Durchsuchen Sie den Hub](./guides/search) nach dem gewünschten Modell oder Datensatz.
- [Greifen Sie auf die Inferenz-API zu](./guides/inference), um schnelle Inferenzen durchzuführen.
