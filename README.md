# Radarfallenerkennung

Implementation of Embedded Yolo to detect speed traps on German roads via camera on a Jetson Nano

Paper: https://www.hindawi.com/journals/mpe/2021/6555513/

## Getting started

- Virtuelle Umgebung erstellen: Im Ordner Radarfallenerkennung/Embedded Yolo/  `conda env create -f environment.yml` in
  einer anaconda shell laufen lassen
- DataAugmentation/README.md lesen
- ggf. SyntheticImages/README.md lesen
- Training / Testing erfolgt in train.py, Beispiele für Kommandozeilenparams sind in der Datei angegeben
- in predictor.py werden in Embedded Yolo/data/samples abgelegte Bilder durch das Netz propagiert und in Embedded
  Yolo/output mit Bounding Boxes abgelegt  