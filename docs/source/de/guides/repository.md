<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Ein Repository erstellen und verwalten

Das Hugging Face Hub besteht aus einer Sammlung von Git-Repositories. [Git](https://git-scm.com/) ist ein in der Softwareentwicklung weit verbreitetes Tool, um Projekte bei der Zusammenarbeit einfach zu versionieren. Dieser Leitfaden zeigt Ihnen, wie Sie mit den Repositories auf dem Hub interagieren, insbesondere:

- Ein Repository erstellen und löschen.
- Zweige (Branches) und Tags verwalten.
- Ihr Repository umbenennen.
- Die Sichtbarkeit Ihres Repositories aktualisieren.
- Eine lokale Kopie Ihres Repositories verwalten.

> [!WARNING]
> Wenn Sie es gewohnt sind, mit Plattformen wie GitLab/GitHub/Bitbucket zu arbeiten, könnte Ihr erster Instinkt sein, die `git` CLI zu verwenden, um Ihr Repo zu klonen (`git clone`), Änderungen zu übernehmen (`git add`, `git commit`) und diese hochzuladen (`git push`). Dies ist beim Verwenden des Hugging Face Hubs gültig. Softwareentwicklung und maschinelles Lernen haben jedoch nicht dieselben Anforderungen und Arbeitsabläufe. Modell-Repositories könnten große Modellgewichtsdateien für verschiedene Frameworks und Tools beinhalten, sodass das Klonen des Repositories dazu führen kann, dass Sie große lokale Ordner mit massiven Größen pflegen. Daher kann es effizienter sein, unsere benutzerdefinierten HTTP-Methoden zu verwenden. Sie können unsere [Git vs HTTP Paradigma](../concepts/git_vs_http) Erklärungsseite für weitere Details lesen.

Wenn Sie ein Repository auf dem Hub erstellen und verwalten möchten, muss Ihr Computer angemeldet sein. Wenn Sie es nicht sind, beziehen Sie sich bitte auf [diesen Abschnitt](../quick-start#login). Im Rest dieses Leitfadens gehen wir davon aus, dass Ihr Computer angemeldet ist.

## Erstellung und Löschung von Repos

Der erste Schritt besteht darin, zu wissen, wie man Repositories erstellt und löscht. Sie können nur Repositories verwalten, die Ihnen gehören (unter Ihrem Benutzernamensraum) oder von Organisationen, in denen Sie Schreibberechtigungen haben.

### Ein Repository erstellen

Erstellen Sie ein leeres Repository mit [`create_repo`] und geben Sie ihm mit dem Parameter `repo_id` einen Namen. Die `repo_id` ist Ihr Namensraum gefolgt vom Repository-Namen: `username_or_org/repo_name`.

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-model")
'https://huggingface.co/lysandre/test-model'
```

Standardmäßig erstellt [`create_repo`] ein Modellrepository. Sie können jedoch den Parameter `repo_type` verwenden, um einen anderen Repository-Typ anzugeben. Wenn Sie beispielsweise ein Dataset-Repository erstellen möchten:

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-dataset", repo_type="dataset")
'https://huggingface.co/datasets/lysandre/test-dataset'
```

Wenn Sie ein Repository erstellen, können Sie mit dem Parameter `private` die Sichtbarkeit Ihres Repositories festlegen.

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-private", private=True)
```

Wenn Sie die Sichtbarkeit des Repositories zu einem späteren Zeitpunkt ändern möchten, können Sie die Funktion [`update_repo_settings`] verwenden.

### Ein Repository löschen

Löschen Sie ein Repository mit [`delete_repo`]. Stellen Sie sicher, dass Sie ein Repository löschen möchten, da dieser Vorgang unwiderruflich ist!

Geben Sie die `repo_id` des Repositories an, das Sie löschen möchten:

```py
>>> delete_repo(repo_id="lysandre/my-corrupted-dataset", repo_type="dataset")
```

### Ein Repository duplizieren (nur für Spaces)

In einigen Fällen möchten Sie möglicherweise das Repo von jemand anderem kopieren, um es an Ihren Anwendungsfall anzupassen. Dies ist für Spaces mit der Methode [`duplicate_space`] möglich. Es wird das gesamte Repository dupliziert.
Sie müssen jedoch noch Ihre eigenen Einstellungen konfigurieren (Hardware, Schlafzeit, Speicher, Variablen und Geheimnisse). Weitere Informationen finden Sie in unserem Leitfaden [Verwalten Ihres Spaces](./manage-spaces).

```py
>>> from huggingface_hub import duplicate_space
>>> duplicate_space("multimodalart/dreambooth-training", private=False)
RepoUrl('https://huggingface.co/spaces/nateraw/dreambooth-training',...)
```

## Dateien hochladen und herunterladen

Jetzt, wo Sie Ihr Repository erstellt haben, möchten Sie Änderungen daran vornehmen und Dateien daraus herunterladen.

Diese 2 Themen verdienen ihre eigenen Leitfäden. Bitte beziehen Sie sich auf die [Hochladen](./upload) und die [Herunterladen](./download) Leitfäden, um zu erfahren, wie Sie Ihr Repository verwenden können.

## Branches und Tags

Git-Repositories verwenden oft Branches, um verschiedene Versionen eines gleichen Repositories zu speichern.
Tags können auch verwendet werden, um einen bestimmten Zustand Ihres Repositories zu kennzeichnen, z. B. bei der Veröffentlichung einer Version.
Allgemeiner gesagt, werden Branches und Tags als [git-Referenzen](https://git-scm.com/book/en/v2/Git-Internals-Git-References) bezeichnet.

### Branches und Tags erstellen

Sie können neue Branches und Tags mit [`create_branch`] und [`create_tag`] erstellen:

```py
>>> from huggingface_hub import create_branch, create_tag

# Erstellen Sie einen Branch auf einem Space-Repo vom `main` Branch
>>> create_branch("Matthijs/speecht5-tts-demo", repo_type="space", branch="handle-dog-speaker")

# Erstellen Sie einen Tag auf einem Dataset-Repo vom `v0.1-release` Branch
>>> create_branch("bigcode/the-stack", repo_type="dataset", revision="v0.1-release", tag="v0.1.1", tag_message="Bump release version.")
```

Sie können die Funktionen [`delete_branch`] und [`delete_tag`] auf die gleiche Weise verwenden, um einen Branch oder einen Tag zu löschen.

### Alle Branches und Tags auflisten

Sie können auch die vorhandenen git-Referenzen von einem Repository mit [`list_repo_refs`] auflisten:

```py
>>> from huggingface_hub import list_repo_refs
>>> list_repo_refs("bigcode/the-stack", repo_type="dataset")
GitRefs(
   branches=[
         GitRefInfo(name='main', ref='refs/heads/main', target_commit='18edc1591d9ce72aa82f56c4431b3c969b210ae3'),
         GitRefInfo(name='v1.1.a1', ref='refs/heads/v1.1.a1', target_commit='f9826b862d1567f3822d3d25649b0d6d22ace714')
   ],
   converts=[],
   tags=[
         GitRefInfo(name='v1.0', ref='refs/tags/v1.0', target_commit='c37a8cd1e382064d8aced5e05543c5f7753834da')
   ]
)
```

## Repository-Einstellungen ändern

Repositories verfügen über einige Einstellungen, die Sie konfigurieren können. Die meiste Zeit möchten Sie dies manuell auf der Repo-Einstellungsseite in Ihrem Browser tun. Sie müssen Schreibzugriff auf ein Repo haben, um es zu konfigurieren (entweder besitzen oder Teil einer Organisation sein). In diesem Abschnitt werden wir die Einstellungen sehen, die Sie auch programmgesteuert mit `huggingface_hub` konfigurieren können.

Einige Einstellungen sind spezifisch für Spaces (Hardware, Umgebungsvariablen,...). Um diese zu konfigurieren, lesen Sie bitte unseren [Verwalten Ihres Spaces](../guides/manage-spaces) Leitfaden.

### Sichtbarkeit aktualisieren

Ein Repository kann öffentlich oder privat sein. Ein privates Repository ist nur für Sie oder die Mitglieder der Organisation sichtbar, in der das Repository sich befindet. Ändern Sie ein Repository wie im Folgenden gezeigt in ein privates:

```py
>>> from huggingface_hub import update_repo_settings
>>> update_repo_settings(repo_id=repo_id, private=True)
```

### Benennen Sie Ihr Repository um

Sie können Ihr Repository auf dem Hub mit [`move_repo] umbenennen. Mit dieser Methode können Sie das Repo auch von einem Benutzer zu einer Organisation verschieben. Dabei gibt es [einige Einschränkungen](https://hf.co/docs/hub/repositories-settings#renaming-or-transferring-a-repo), die Sie beachten sollten. Zum Beispiel können Sie Ihr Repo nicht an einen anderen Benutzer übertragen.

```py
>>> from huggingface_hub import move_repo
>>> move_repo(from_id="Wauplin/cool-model", to_id="huggingface/cool-model")
```

## Verwalten Sie eine lokale Kopie Ihres Repositories

Alle oben beschriebenen Aktionen können mit HTTP-Anfragen durchgeführt werden. In einigen Fällen möchten Sie jedoch vielleicht eine lokale Kopie Ihres Repositories haben und damit interagieren, indem Sie die Git-Befehle verwenden, die Sie kennen.

Die [`Repository`] Klasse ermöglicht es Ihnen, mit Dateien und Repositories auf dem Hub mit Funktionen zu interagieren, die Git-Befehlen ähneln. Es ist ein Wrapper über Git und Git-LFS-Methoden, um die Git-Befehle zu verwenden, die Sie bereits kennen und lieben. Stellen Sie vor dem Start sicher, dass Sie Git-LFS installiert haben (siehe [hier](https://git-lfs.github.com/) für Installationsanweisungen).

### Verwenden eines lokalen Repositories

Instanziieren Sie ein [`Repository`] Objekt mit einem Pfad zu einem lokalen Repository:

```py
>>> from huggingface_hub import Repository
>>> repo = Repository(local_dir="<path>/<to>/<folder>")
```

### Klonen

Der `clone_from` Parameter klont ein Repository von einer Hugging Face Repository-ID in ein lokales Verzeichnis, das durch das Argument `local_dir` angegeben wird:

```py
>>> from huggingface_hub import Repository
>>> repo = Repository(local_dir="w2v2", clone_from="facebook/wav2vec2-large-960h-lv60")
```

`clone_from` kann auch ein Repository mit einer URL klonen:

```py
>>> repo = Repository(local_dir="huggingface-hub", clone_from="https://huggingface.co/facebook/wav2vec2-large-960h-lv60")
```

Sie können den `clone_from` Parameter mit [`create_repo`] kombinieren, um ein Repository zu erstellen und zu klonen:

```py
>>> repo_url = create_repo(repo_id="repo_name")
>>> repo = Repository(local_dir="repo_local_path", clone_from=repo_url)
```

Sie können auch einen Git-Benutzernamen und eine E-Mail zu einem geklonten Repository konfigurieren, indem Sie die Parameter `git_user` und `git_email` beim Klonen eines Repositories angeben. Wenn Benutzer Änderungen in diesem Repository committen, wird Git über den Autor des Commits informiert sein.

```py
>>> repo = Repository(
...   "my-dataset",
...   clone_from="<user>/<dataset_id>",
...   token=True,
...   repo_type="dataset",
...   git_user="MyName",
...   git_email="me@cool.mail"
... )
```

### Branch

Branches sind wichtig für die Zusammenarbeit und das Experimentieren, ohne Ihre aktuellen Dateien und Codes zu beeinflussen. Wechseln Sie zwischen den Branches mit [`~Repository.git_checkout`]. Wenn Sie beispielsweise von `branch1` zu `branch2` wechseln möchten:

```py
>>> from huggingface_hub import Repository
>>> repo = Repository(local_dir="huggingface-hub", clone_from="<user>/<dataset_id>", revision='branch1')
>>> repo.git_checkout("branch2")
```

### Pull

Mit [`~Repository.git_pull`] können Sie eine aktuelle lokale Branch mit Änderungen aus einem Remote-Repository aktualisieren:

```py
>>> from huggingface_hub import Repository
>>> repo.git_pull()
```

Setzen Sie `rebase=True`, wenn Sie möchten, dass Ihre lokalen Commits nach dem Aktualisieren Ihres Zweigs mit den neuen Commits aus dem Remote erfolgen:

```py
>>> repo.git_pull(rebase=True)
```
