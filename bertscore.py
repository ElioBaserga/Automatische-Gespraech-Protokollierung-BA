import json
import os
from bert_score import score

generated_file = "generatedProtocols/GPT-4o-mini/Akanji_GPT-4o-mini.json"
reference_file = "referenceProtocols/Akanji-Original-Protocol.json"

# JSON-Dateien einlesen
with open(generated_file, 'r', encoding='utf-8') as f:
    generated_data = json.load(f)

with open(reference_file, 'r', encoding='utf-8') as f:
    reference_data = json.load(f)

# Antworten extrahieren
candidates = [item["antwort"] for item in generated_data]
references = [item["antwort"] for item in reference_data]

# BERTScore berechnen
P, R, F1 = score(candidates, references, lang="de", verbose=False)

for i, (p, r, f) in enumerate(zip(P, R, F1), 1):
    print(f"Frage {i} -> Precision: {p.item():.4f} | Recall: {r.item():.4f} | F1-Score: {f.item():.4f}")

# Den Gesamtdurchschnitt über alle Fragen ausgeben
print("\nGesamtdurchschnitt")
print(f"Mean Precision: {P.mean().item():.4f} | Mean Recall: {R.mean().item():.4f} | Mean F1-Score: {F1.mean().item():.4f}")