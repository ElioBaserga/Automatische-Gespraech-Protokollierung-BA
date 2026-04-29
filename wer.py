from jiwer import wer
import jiwer

ref_file = "referenceTranscripts/Akanji-reference-transcript.txt"
gen_file = "generatedTranscripts/Akanji-generated-transcript.txt"

# Dateien einlesen
with open(ref_file, "r", encoding="utf-8") as r, open(gen_file, "r", encoding="utf-8") as g:
    referenz = r.read()
    generated = g.read()

# Bereinigung
bereinigung = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])

ref_bereingt = bereinigung(referenz)
gen_bereingt = bereinigung(generated)

# WER berechnen
fehlerquote = wer(ref_bereingt, gen_bereingt)

print(f"{fehlerquote:.2%}")