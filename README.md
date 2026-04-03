---
title: MindMirror
emoji: 🪞
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "6.9.0"
app_file: app.py
pinned: false
license: mit
---

# 🪞 MindMirror — AI Interview Coach

Real-time behavioral coaching that watches your face and listens to your voice
during interview practice, then delivers live nudges and a scored report.

## How it works

1. Allow camera and microphone access in your browser
2. Activate the microphone input (click the mic button)
3. Enter your interview question and click **Start Session**
4. MindMirror calibrates your baseline for the first ~15 seconds
5. Live coaching nudges appear as you speak
6. Click **End Session** to receive your full coaching report

## Scoring

Each session produces two scores combined multiplicatively:

| Score | What it measures |
|-------|-----------------|
| **Delivery** | Eye contact, confident state %, filler words, speaking rate |
| **Content** | Did you actually answer the question asked? |
| **Final** | Delivery × Content — an off-topic answer scores near 0 regardless of delivery |

## HF Spaces setup (admins)

Add these Space secrets before launching:

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | HuggingFace token with **write** access to the sessions dataset |
| `HF_DATASET_REPO` | Dataset repo ID, e.g. `your-username/mindmirror-sessions` |
| `LLM_PROVIDER` | Optional. Defaults to `hf` (free HF Inference API). Set to `gemini` if you have a Gemini API key. |
| `GEMINI_API_KEY` | Only needed if `LLM_PROVIDER=gemini` |

Create a **private** HF Dataset repo for session storage before first launch.
Users log in with their HuggingFace accounts — no separate password needed.

## Local development

```bash
git clone <repo>
cd mindmirror
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # fill in GEMINI_API_KEY (or use LLM_PROVIDER=ollama)
ollama pull llama3.2      # if using ollama
python app.py
```

Open http://localhost:7860. Enter your name when prompted (no login needed locally).
