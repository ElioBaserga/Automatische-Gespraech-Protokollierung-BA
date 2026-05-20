import json
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

def lade_json(dateipfad):
    annotation = Annotation()
    
    # JSON-Datei öffnen und einlesen
    with open(dateipfad, "r", encoding="utf-8") as f:
        daten = json.load(f)
        
        # Jedes Element aus dem JSON in die pyannote-Annotation eintragen
        for eintrag in daten:
            segment = Segment(eintrag["start"], eintrag["ende"])
            # sieht am Schluss so aus: {"sprecher": "Sprecher_A", "start": 0.0, "ende": 5.5}
            annotation[segment] = eintrag["sprecher"]
            
    return annotation

# Dateien in pyannote-Objekte umwandeln
reference_file = lade_json("referenceDiarization/Dolmetscher-reference-diarization.json")
generated_file = lade_json("generatedDiarization/Dolmetscher-generated-diarization.json")

# collar in Sekunden, um kleine zeitliche Abweichungen zu tolerieren
der_metrik = DiarizationErrorRate(collar=0.5)
fehlerquote = der_metrik(reference_file, generated_file)

print(f"{fehlerquote:.2%}")