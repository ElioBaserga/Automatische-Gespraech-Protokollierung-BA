import streamlit as st
import whisper
import torch
import librosa
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from openai import OpenAI
import os
from dotenv import load_dotenv
import tempfile
import json

# Wird benötigt, damit nach dem JSON-Download die Seite nicht komplett neu lädt und die vorherige Ausgabe verschwindet
if "llm_answer" not in st.session_state:
    st.session_state.llm_answer = None

uploaded_file = st.file_uploader("Upload Gespräch:")

user_question = st.text_area("Fragen:")

llm_choice = st.selectbox("Wähle das Sprachmodell:", ["GPT-4o-mini", "Meta Llama 3 8B", "DeepSeek R1"])

if st.button("Start Protokollierung"):
    if uploaded_file is not None and user_question.strip():

        # Ladebalken
        progress_bar = st.progress(0)

        # Hochgeladene Datei kurz zwischenspeichern, damit Whisper/librosa einen Dateipfad haben
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            audio_file = tmp_file.name

        progress_bar.progress(5)

        load_dotenv()

        hf_token = os.getenv("HF_TOKEN")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        # Whisper Modell laden: es gibt: tiny, base, small, medium, large -> Je grösser, desto mehr VRAM wird benötigt, aber dafür genauer.
        model = whisper.load_model("large")
        progress_bar.progress(20)

        # Zu HuggingFace verbinden, um die Pyannote Pipeline zu laden (Sprecher-Diarization)
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        progress_bar.progress(40)

        # Transkription mit Whisper
        result = model.transcribe(audio_file)
        progress_bar.progress(60)

        # librosa load um die Audiodatei zu laden; pyannote arbeitet mit 16kHz, darum sr=16000
        # (16000 mal pro Sekunde abtasten der Lautstärke); waveform, um die Audiowellen zu speichern
        waveform, sample_rate = librosa.load(audio_file, sr=16000)

        # Audiodatei für das LLM vorbereiten: pyannote erwartet einen Tensor; Tensor=Container für Zahlen, der extrem schnell verarbeitet werden kann
        waveform_tensor = torch.tensor(waveform).unsqueeze(0)

        # Audiodatei pyannote übergeben für die Speaker Diarization
        diarization_result = pipeline({"waveform": waveform_tensor, "sample_rate": sample_rate})
        progress_bar.progress(80)

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
            # print(line)
            transcript_lines.append(line)

        # Liste zu einem einzigen String zusammenfügen
        full_transcript = "\n".join(transcript_lines)
        progress_bar.progress(90)

        # LLM Abfrage (user_question kommt jetzt aus dem Streamlit text_area)
        
        def ask_llm(transcript: str, question: str, api_key: str, choice: str) -> str:

            if choice == "GPT-4o-mini":
                client = OpenAI(api_key=api_key)
                model_name = "gpt-4o-mini"
            elif choice == "Meta Llama 3 8B":
                # Die OpenAI Library wird verwendet, um ollama lokal aufzurufen
                client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
                model_name = "llama3"
            elif choice == "DeepSeek R1":
                # DeepSeek R1 wird auch über ollama lokal aufgerufen
                client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
                model_name = "deepseek-r1"

            # Prompt zwingt das LLM, JSON zu generieren
            prompt = f"""Hier ist das Transkript eines Interviews:\n\n{transcript}\n\n
            Das Transkript ist ein Gespräch zwischen einer Person, die Fragen stellt und eine Person, die antwortet.\n\n
            Beantworte bitte die folgenden Fragen jeweils mit einem ganzen Satz: {question}\n\n
            WICHTIGE REGELN FÜR DIE AUSGABE:
            1. Du MUSST ein valides JSON-Array zurückgeben.
            2. Verwende NIEMALS Markdown-Formatierung (wie ```json).
            3. Beginne deine Antwort direkt mit [ und ende direkt mit ].
            4. Schreibe keinen erklärenden Text davor oder danach.
            
            Format:
            [
                {{"frage": "Die erste Frage", "antwort": "Die Antwort dazu"}},
                {{"frage": "Die zweite Frage", "antwort": "Die Antwort dazu"}}
            ]"""

            # Maximal 3 Versuche um ein gültiges JSON zu erstellen
            for versuch in range(3):
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        # role: system: Leitplanken für die KI; role: user: die eigentliche Frage, die an die KI gestellt wird
                        messages=[
                            {"role": "system", "content": "Du bist ein Assistent, der Fragen zu Audio-Transkripten beantwortet und ausschließlich JSON ausgibt."},
                            {"role": "user", "content": prompt}
                        ],
                        # 0.7 ist ein guter Wert, um eine kreative Antwort zu bekommen, aber nicht zu verrückt; je höher, desto kreativer, je niedriger, desto fokussierter auf die Fakten
                        temperature=0.7
                    )
                    # Die Antwort der KI auslesen
                    ans = response.choices[0].message.content.strip()
                    
                    # prüfen, ob das Resultat gültiges JSON ist
                    json.loads(ans) 
                    return ans
                except Exception as e:
                    print(f"Versuch {versuch+1} gescheitert. LLM hat kein reines JSON geliefert.")
                    continue
            return '[{"frage": "Systemfehler", "antwort": "Das LLM konnte nach 3 Versuchen kein gültiges JSON generieren."}]'
            
        llm_answer = ask_llm(full_transcript, user_question, openai_api_key, llm_choice)

        # Antwort im Speicher ablegen, damit sie auch nach einem Seiten-Reload noch verfügbar ist
        st.session_state.llm_answer = llm_answer

        progress_bar.progress(100)

        # Temporäre Datei löschen
        if os.path.exists(audio_file):
            os.remove(audio_file)

if st.session_state.llm_answer is not None:
    answer_data = json.loads(st.session_state.llm_answer)
        
    for item in answer_data:
        st.write(f"Frage: {item.get('frage', '')}")
        st.write(f"Antwort: {item.get('antwort', '')}")
        st.write("") # leere Zeile

    st.download_button(
        label="Download JSON",
        data=st.session_state.llm_answer,
        file_name="llm_output.json",
        mime="application/json"
    )