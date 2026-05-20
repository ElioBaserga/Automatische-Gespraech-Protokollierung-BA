from jiwer import wer
import jiwer

reference_file = "referenceTranscripts/Dolmetscher-reference-transcript.txt"
generated_file = "generatedTranscripts/Dolmetscher-generated-transcript.txt"

# Dateien einlesen
with open(reference_file, "r", encoding="utf-8") as r, open(generated_file, "r", encoding="utf-8") as g:
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