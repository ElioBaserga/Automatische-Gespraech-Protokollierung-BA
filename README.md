# Automatische Gesprächsprotokollierung - Prototyp

## Übersicht

Dieser Prototyp wurde im Rahmen einer Bachelorarbeit entwickelt und bietet eine automatisierte Pipeline zur Erstellung von Gesprächsprotokollen. Der primäre Fokus liegt auf sensiblen Behördengesprächen, welche auf Schweizerdeutsch gehalten werden. Das Ziel ist es, den zeitaufwendigen Prozess der manuellen Protokollierung durch den Einsatz von Large Language Models und weiteren KI-Technologien zu automatisieren, ohne dabei den Datenschutz zu vernachlässigen.

## Hauptfunktionen
- Transkription: Automatische Umwandlung von Audio in hochdeutschen Text mittels OpenAI Whisper.
- Sprecherdiarisierung: Automatische Erkennung und Zuweisung von Sprecherwechseln mithilfe von PyAnnote.
- Datenschutzkonforme Informationsextraktion: Beantwortung spezifischer Protokollfragen durch lokal über Ollama gehostete Sprachmodelle.
- Wissenschaftliche Evaluation: Integrierte Python-Skripte zur Berechnung der Word Error Rate, der Diarization Error Rate und des BERT-Scores.

## Setup

### Voraussetzungen

- Python 3.10.19
- FFmpeg
- Ollama + LLMs:
  - Meta Llama 3
  - DeepSeek-R1
  - Google Gemma 4
  - Optional: GPT-4o-mini

- Hugging Face Account & Token
- PyAnnote Bedingungen akzeptieren:
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

##### Fundament installieren:
```bash
pip install pyannote.audio torch torchvision torchaudio
```

##### Weitere Packages:
```bash
pip install openai-whisper streamlit librosa openai python-dotenv bert-score jiwer pyannote.metrics
```

#### .env Datei erstellen:
```properties
HF_TOKEN=huggingface_token_einfuegen

# optional:
OPENAI_API_KEY=openai_key_einfuegen
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

- Arbeitsplan: Im Ordner *doc* ist der Arbeitsplan zu finden.
- Meeting-Protokolle: Im Unterordner *doc/meetings* befinden sich die Protokolle der Besprechungen mit dem Betreuer dieser Arbeit, Benjamin Kühnis.
- Projektmanagement: In der Lasche *Projects* auf diesem GitHub-Repository ist das Kanban-Board für dieses Projekt zu finden.
