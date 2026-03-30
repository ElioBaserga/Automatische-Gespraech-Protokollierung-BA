import json
import os
from bert_score import score

generated_file = "generatedProtocols/chatgpt/llm_output_3.json"
reference_file = "referenceProtocols/Roger_Federer-Interview-Original-Protocol.json"

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
    print(f"Frage {i} -> Precision: {p:.4f} | Recall: {r:.4f} | F1-Score: {f:.4f}")