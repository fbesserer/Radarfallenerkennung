### 1. Trainingsbilder als background Bilder hernehmen (44000 Stück)

- aus diesen wird die benötigte Anzahl an Bildern (+ weitere Pufferbilder) zufällig ausgewählt,
- in queue gespeichert und bei bedarf geladen

### 2. random erase in backgrounds einfügen, so dass die bereits existierende bounding box maximal zu 75 % überlagert wird (occlusion)

- falls nicht möglich, bild discarden und eines der pufferbilder laden [edge case]
- Bildebearbeitung kann parallel erfolgen. Bei Speicherung der fertigen Bilder muss synchronisiert werden
- Speicherung erfolgt entweder in eigener Klasse oder in spezieller concurrency list

### 3. parallel die foreground blitzer & distractor objects skalieren und unterschiedlich augmentieren

- Jedes foreground Objekt wird als eigenes Objekt gespeichert mit Attributen wie Blitzerklasse etc
- Nach Nutzen die jeweiligen Objekte aus den Datenstrukturen entfernen

### 4. fg und bg zusammenfügen

- (3 x mit unterchiedlichem Blending), dabei in 20% der
  Fälle Occlusion/Truncationprovozieren (Truncation: >= 25% der Box im Bild,
  Occlusion: max IOU 75%), in 50% der Bilder Distractor Objects einfügen, Ergebnisse
  loggen und statistisch asuwerten kompletten vorgang mitloggen, so dass im nachhinein
  nachvollzogen werden kann, für welches Bild welche Entscheidungen getroffen wurden.

### 1 Executor für backgrounds

### 1 Executor für foregrounds

### auf beide mit join warten und danach bei 4 weiter im Main Thread der alles übergeordnet steuert