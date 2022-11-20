## augment_images.py

- Trainings-, Valid- und Testbilder müssen mit entsprechenden txt files vorhanden sein
- Skript starten. Einen der og Ordner auswählen. Zielverzeichnis wählen in das die augmentierten Bilder abgelegt werden
  sollen.

Wenn bereits augmentierte Bilder vorliegen kann dieser Schritt übersprungen werden. Das Skript darknet_prerequisites.py
muss zur Anpassung der Pfade in jedem Fall ausgeführt werden.

## darknet_prerequisites.py

Erstellt aus den .txt annotation files der Bilder die für das Training notwendige kombinierte .txt Datei, die die Pfade
aller entsprechenden Bilder enthält.

für Trainings-, Valid- und Testbilder jeweils

- globale Variable TXT_NAME anpassen
- Skript starten
- entstandene xxxx.txt in Radarfallenerkennung\Embedded Yolo\data\ ablegen 
