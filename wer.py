from jiwer import wer

ref_file = "referenceTranscript/Akanji-Original-Transcript.txt"
gen_file = "generatedTranscript/Akanji-Generated-Transcript.txt"

# Inhalte aus den Dateien lesen
with open(ref_file, "r", encoding="utf-8") as r, open(gen_file, "r", encoding="utf-8") as g:
    referenz = r.read()
    generated = g.read()

# WER berechnen
fehlerquote = wer(referenz, generated)

print(fehlerquote)