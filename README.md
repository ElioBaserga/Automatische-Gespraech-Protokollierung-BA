# Automatische-Gespräch-Protokollierung-BA

## Setup

### Voraussetzungen

- Python 3.10.19
- FFmpeg
- Ollama + LLMs:
  - Meta Llama 3
  - DeepSeek-R1
  - Google Gemma 4

- Hugging Face Account & Token
- PyAnnote Bedinungen Akzeptieren:
  - pyannote/speaker-diarization-3.1
  - pyannote/segmentation-3.0

### Installation

#### Virtuelle Umgebung erstellen:
```bash
python -m venv venv
```
```bash
venv\Scripts\activate
```

#### Abhängigkeiten installieren:
```bash
pip install streamlit librosa pyannote.audio openai python-dotenv bert-score jiwer pyannote.metrics
```

##### Whisper installieren für die..
CPU:
```bash
pip install openai-whisper
```
GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
```bash
pip install openai-whisper
```
<br>

#### .env Datei erstellen:
```properties
HF_TOKEN=huggingface_token_einfügen
```

## Nutzung

### Applikation starten

Die grafische Benutzeroberfläche wird mit folgendem Befehl ausgeführt:
```bash
streamlit run app.py
```

### Ablauf in der App

1. Eine Audiodatei des Gesprächs hochladen.
2. Im Textfeld die Fragen eingeben, die das LLM aus dem Gespräch beantworten soll.
3. Das Sprachmodell auswählen, das die Fragen beantwortet (Meta Llama 3, DeepSeek R1, Google Gemma 4).
4. Auf "Start Protokollierung" klicken.

## Dokumentation

Im Ordner "doc" ist einerseits der Arbeitsplan zu finden und andererseits im Unterordner "meetings" die Protokolle der Besprechungen mit dem Betreuer dieser Arbeit, Benjamin Kühnis.

In der Lasche Projects ist das Kanban-Board für dieses Projekt zu finden.
