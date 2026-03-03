import whisper
import torch
import librosa
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from openai import OpenAI
import os
from dotenv import load_dotenv

audio_file = "Demo-Interview-Short.mp3"

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Whisper Modell laden: es gibt: tiny, base, small, medium, large -> Je grösser, desto mehr VRAM wird benötigt, aber dafür genauer.
model = whisper.load_model("large")

# Zu HuggingFace verbinden, um die Pyannote Pipeline zu laden (Sprecher-Diarization)
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=hf_token
)

# Transkription mit Whisper
result = model.transcribe(audio_file)

# librosa load um die Audiodatei zu laden; pyannote arbeitet mit 16kHz, darum sr=16000
# (16000 mal pro Sekunde abtasten der Lautstärke); waveform, um die Audiowellen zu speichern
waveform, sample_rate = librosa.load(audio_file, sr=16000)

# Audiodatei für das LLM vorbereiten: pyannote erwartet einen Tensor; Tensor=Container für Zahlen, der extrem schnell verarbeitet werden kann
waveform_tensor = torch.tensor(waveform).unsqueeze(0)

# Audiodatei pyannote übergeben für die Speaker Diarization
diarization_result = pipeline({"waveform": waveform_tensor, "sample_rate": sample_rate})

# Hatte Probleme mit dem Ouput von pyannote, darum hier die verschiedenen Möglichkeiten abfangen, wie die Diarization zurückgegeben werden könnte
diarization = None

if hasattr(diarization_result, "speaker_diarization"):
    diarization = diarization_result.speaker_diarization
elif isinstance(diarization_result, Annotation):
    diarization = diarization_result
elif isinstance(diarization_result, tuple):
    diarization = diarization_result[0]

if diarization is None:
    diarization = diarization_result

tracks = []
# yield_label=True macht, dass die Sprecher Labels auch angegeben werden (SPEAKER_00 / SPEAKER_01)
# itertracks gibt die Zeitintervalle zurück
tracks = list(diarization.itertracks(yield_label=True))

# Wir speichern das gesamte Transkript in einer Liste, um es später an OpenAI zu senden
transcript_lines = []

# Segmente von Whisper durchgehen und die Sprecherinformationen von pyannote danach anhängen
for segment in result["segments"]:
    start = segment["start"]
    end = segment["end"]
    text = segment["text"].strip()
    
    # Wir nehmen die Mitte des Segments, um den Sprecher zu bestimmen, um aus pyannote und whisper den gleichen Zeitpunkt zu vergleichen
    mid_time = (start + end) / 2
    speaker = "Unbekannt"
    
    # tracks dürfen nicht leer sein
    if tracks:
        # pyannote Zeitintervalle durchgehen, um den Sprecher zu finden, der zum Zeitpunkt des Segments spricht
        for turn, _, speaker_label in tracks:
            if turn.start <= mid_time <= turn.end:
                speaker = speaker_label
                break
            
    line = f"[{start:5.2f}s - {end:5.2f}s] {speaker}: {text}"
    print(line)
    transcript_lines.append(line)

# Liste zu einem einzigen String zusammenfügen
full_transcript = "\n".join(transcript_lines)

# LLM Abfrage
user_question = "Wer ist die Hauptperson im Interview und über was redet er?"

def ask_openai(transcript: str, question: str, api_key: str) -> str:
    client = OpenAI(api_key=api_key)

    prompt = f"""Hier ist das Transkript eines Interviews:\n\n{transcript}\n\nBasierend auf diesem Transkript, beantworte bitte folgende Frage:\n{question}"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            # role: system: Leitplanken für die KI; role: user: die eigentliche Frage, die an die KI gestellt wird
            messages=[
                {"role": "system", "content": "Du bist ein Assistent, der Fragen zu Audio-Transkripten beantwortet."},
                {"role": "user", "content": prompt}
            ],
            # 0.7 ist ein guter Wert, um eine kreative Antwort zu bekommen, aber nicht zu verrückt; je höher, desto kreativer, je niedriger, desto fokussierter auf die Fakten
            temperature=0.7
        )
        # Die Antwort der KI zurückgeben
        return response.choices[0].message.content
    except Exception as e:
        return f"Fehler bei der OpenAI-API-Anfrage: {e}"
    
llm_answer = ask_openai(full_transcript, user_question, openai_api_key)

print(llm_answer)