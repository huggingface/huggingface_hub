<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Git vs. HTTP-Paradigma

Die `huggingface_hub`-Bibliothek ist eine Bibliothek zur Interaktion mit dem Hugging Face
Hub, einer Sammlung von auf Git basierenden Repositories (Modelle, Datensätze oder
Spaces). Es gibt zwei Hauptmethoden, um auf den Hub mit `huggingface_hub` zuzugreifen.

Der erste Ansatz, der sogenannte "Git-basierte" Ansatz, wird von der [`Repository`] Klasse
geleitet. Diese Methode verwendet einen Wrapper um den `git`-Befehl mit zusätzlichen
Funktionen, die speziell für die Interaktion mit dem Hub entwickelt wurden. Die zweite
Option, die als "HTTP-basierter" Ansatz bezeichnet wird, umfasst das Senden von
HTTP-Anfragen mit dem [`HfApi`] Client. Schauen wir uns die Vor- und Nachteile jeder
Methode an.

## Repository: Der historische git-basierte Ansatz

Ursprünglich wurde `huggingface_hub` größtenteils um die [`Repository`] Klasse herum
entwickelt. Sie bietet Python-Wrapper für gängige git-Befehle wie `"git add"`, `"git commit"`,
`"git push"`, `"git tag"`, `"git checkout"` usw.

Die Bibliothek hilft auch beim Festlegen von Zugangsdaten und beim Tracking von großen
Dateien, die in Machine-Learning-Repositories häufig verwendet werden. Darüber hinaus
ermöglicht die Bibliothek das Ausführen ihrer Methoden im Hintergrund, was nützlich ist,
um Daten während des Trainings hochzuladen.

Der Hauptvorteil bei der Verwendung einer [`Repository`] besteht darin, dass Sie eine
lokale Kopie des gesamten Repositorys auf Ihrem Computer pflegen können. Dies kann jedoch
auch ein Nachteil sein, da es erfordert, diese lokale Kopie ständig zu aktualisieren und
zu pflegen. Dies ähnelt der traditionellen Softwareentwicklung, bei der jeder Entwickler
eine eigene lokale Kopie pflegt und Änderungen überträgt, wenn an einer Funktion
gearbeitet wird. Im Kontext des Machine Learning ist dies jedoch nicht immer erforderlich,
da Benutzer möglicherweise nur Gewichte für die Inferenz herunterladen oder Gewichte von
einem Format in ein anderes konvertieren müssen, ohne das gesamte Repository zu klonen.

## HfApi: Ein flexibler und praktischer HTTP-Client

Die [`HfApi`] Klasse wurde entwickelt, um eine Alternative zu lokalen Git-Repositories
bereitzustellen, die besonders bei der Arbeit mit großen Modellen oder Datensätzen
umständlich zu pflegen sein können. Die [`HfApi`] Klasse bietet die gleiche Funktionalität
wie git-basierte Ansätze, wie das Herunterladen und Hochladen von Dateien sowie das
Erstellen von Branches und Tags, jedoch ohne die Notwendigkeit eines lokalen Ordners, der
synchronisiert werden muss.

Zusätzlich zu den bereits von `git` bereitgestellten Funktionen bietet die [`HfApi`]
Klasse zusätzliche Features wie die Möglichkeit, Repositories zu verwalten, Dateien mit
Caching für effiziente Wiederverwendung herunterzuladen, im Hub nach Repositories und
Metadaten zu suchen, auf Community-Funktionen wie Diskussionen, Pull Requests und
Kommentare zuzugreifen und Spaces-Hardware und Geheimnisse zu konfigurieren.

## Was sollte ich verwenden ? Und wann ?

Insgesamt ist der **HTTP-basierte Ansatz in den meisten Fällen die empfohlene Methode zur Verwendung von**
`huggingface_hub`. Es gibt jedoch einige Situationen, in denen es vorteilhaft sein kann,
eine lokale Git-Kopie (mit [`Repository`]) zu pflegen:
- Wenn Sie ein Modell auf Ihrem Computer trainieren, kann es effizienter sein, einen
herkömmlichen git-basierten Workflow zu verwenden und regelmäßige Updates zu pushen.
[`Repository`] ist für diese Art von Situation mit seiner Fähigkeit zur Hintergrundarbeit optimiert.
- Wenn Sie große Dateien manuell bearbeiten müssen, ist `git` die beste Option, da es nur
die Differenz an den Server sendet. Mit dem [`HfAPI`] Client wird die gesamte Datei bei
jeder Bearbeitung hochgeladen. Beachten Sie jedoch, dass die meisten großen Dateien binär
sind und daher sowieso nicht von Git-Diffs profitieren.

Nicht alle Git-Befehle sind über [`HfApi`] verfügbar. Einige werden vielleicht nie
implementiert, aber wir bemühen uns ständig, die Lücken zu schließen und zu verbessern.
Wenn Sie Ihren Anwendungsfall nicht abgedeckt sehen, öffnen Sie bitte [ein Issue auf
Github](https://github.com/huggingface/huggingface_hub)! Wir freuen uns über Feedback, um das 🤗-Ökosystem mit und für unsere Benutzer aufzubauen.
