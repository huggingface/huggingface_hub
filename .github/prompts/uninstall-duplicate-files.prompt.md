---
name: "Duplikate finden und entfernen"
description: "Findet exakt gleiche Dateien im Workspace und entfernt bestätigte Duplikate."
argument-hint: "Optionaler Ordnerpfad zum Prüfen"
agent: "agent"
---

Finde exakte doppelte Dateien im aktuellen Workspace. Nutze den angegebenen Ordner als Suchbereich, falls einer genannt wurde; andernfalls prüfe den Workspace. Bleibe innerhalb dieses Bereichs.

Vergleiche Dateien anhand ihres Inhalts, nicht anhand ihrer Namen. Verwende Hashes zum Auffinden möglicher Treffer und vergleiche die Bytes vor der Einstufung nochmals direkt. Ähnlich benannte oder nur teilweise ähnliche Dateien sind keine Duplikate. Wenn du Inhalte nicht zuverlässig vergleichen kannst, melde das, statt eine Datei als Duplikat einzustufen.

Überspringe `.git`, virtuelle Umgebungen, Abhängigkeitsverzeichnisse, Caches und Build-Ausgaben. Führe keine Dateien außerhalb des Workspaces auf und folge keinen Symlinks aus dem Suchbereich heraus.

Berichte zunächst nur die gefundenen Gruppen. Nenne je Gruppe die Pfade, Dateigrößen und einen empfohlenen Pfad zum Behalten. Bevorzuge dabei eine versionierte, kanonisch platzierte Datei; lösche oder überschreibe nichts automatisch.

Warte auf die ausdrückliche Bestätigung, welche Pfade entfernt werden dürfen. Entferne danach ausschließlich die bestätigten Dateien, behalte mindestens eine Datei je Duplikatgruppe und prüfe anschließend, dass die behaltenen Dateien weiterhin vorhanden sind. Falls die Bestätigung unklar ist oder sich der Workspace seit dem Bericht geändert hat, frage nach beziehungsweise untersuche die betroffenen Dateien erneut.

Gib das Ergebnis knapp auf Deutsch aus: geprüfter Bereich, gefundene Duplikatgruppen, vorgeschlagene Behalte-Pfade und nach Bestätigung tatsächlich entfernte Dateien.