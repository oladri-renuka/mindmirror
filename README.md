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

<div align="center">

# 🪞 MindMirror

### Real-Time AI Interview Coach — Multimodal Behavioral Analysis

[![Live Demo](https://img.shields.io/badge/🤗%20HF%20Spaces-Live%20Demo-blue)](https://huggingface.co/spaces/oladri-Renuka/mindmirror)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-6.9-orange?logo=gradio)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

MindMirror watches your face and listens to your voice in real time during interview practice. It detects behavioral signals — confidence, nervousness, eye contact, filler words — and delivers live AI coaching nudges every 2 seconds, then generates a scored report with specific improvement advice.

</div>

---

## What it does

| Signal | How it's captured |
|--------|------------------|
| **Eye contact** | MediaPipe 478-point face mesh → gaze vector projection |
| **Facial state** | Eye Aspect Ratio, Mouth Aspect Ratio, brow stress, head pose (yaw/pitch) |
| **Speech** | Web Speech API (Chrome/Edge) with Whisper fallback — real-time transcription |
| **Vocal patterns** | Pitch (YIN algorithm), speaking rate (WPM), pause detection |
| **Filler/hedge words** | Token-level detection: "um", "uh", "like", "you know", "kind of", etc. |

These signals are fused every 2 seconds into a **behavioral state** — one of: `CONFIDENT`, `NERVOUS`, `UNCERTAIN`, `THINKING`, `DISENGAGED`, or `NEUTRAL` — calibrated against your personal baseline from the first 15 seconds.

A LangGraph agent monitors state over time and fires coaching nudges when negative patterns persist. At session end it generates a structured 5-section coaching report scored across delivery and content.

---

## Architecture

```
Browser
  ├── Webcam frames (7–10 fps)   ──▶  MediaPipe FaceLandmarker
  │                                        │
  │                                   FacialFeatureVector
  │                                        │
  └── Microphone audio                     ▼
        │                            Feature Fusion (2s window)
        ├── Web Speech API  ──────▶        │
        └── Whisper fallback               ▼
                                    PersonalBaseline calibration
                                           │
                                    BehavioralStateClassifier
                                           │
                                    LangGraph Agent ──▶ Nudge / Report
                                           │
                                    SessionLogger (HF Dataset)
```

**Key design decisions:**
- **Browser-mode pipeline** — no server-side mic or webcam threads; Gradio stream callbacks push data in. Works on HF Spaces out of the box.
- **Personal baseline** — first 7 windows (~15s) establish your own resting state, so the classifier is relative to *you*, not a global average.
- **Multiplicative scoring** — `Final = (Delivery/10) × (Content/10) × 10`. An off-topic answer scores near zero regardless of how confidently it was delivered.
- **LLM fallback chain** — Gemini 2.0 Flash → HF Inference API → rule-based. No single point of failure.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Gradio 6.9, Web Speech API (JS) |
| Vision | MediaPipe 0.10.33, OpenCV |
| Speech | faster-whisper (base, int8), Web Speech API |
| Audio features | librosa, YIN pitch, custom filler/hedge detector |
| Agent | LangGraph 1.1.3, LangChain Core |
| LLM | Google Gemini 2.0 Flash / HF Inference API |
| Data | Pydantic v2, NumPy, Pandas, scikit-learn |
| Storage | HuggingFace Datasets (cloud) / local JSON (dev) |

---

## Performance

| Component | Latency | Hardware |
|-----------|---------|----------|
| MediaPipe face mesh | ~8 ms/frame | Apple M4 |
| faster-whisper base (2s chunk) | ~350 ms | Apple M4 CPU |
| Behavioral classification | < 1 ms | — |
| LangGraph nudge generation | ~800 ms | Gemini API |
| Full pipeline tick | ~1.2 s | Apple M4 |

The 2-second timer fires well within the processing budget. Web Speech API delivers transcripts in < 300 ms on Chrome.

---

## Project Structure

```
mindmirror/
├── app.py                        # Gradio UI + JS Web Speech bridge
├── pipeline.py                   # Session orchestrator
├── src/
│   ├── vision/
│   │   ├── face_tracker.py       # MediaPipe FaceLandmarker wrapper
│   │   └── facial_features.py    # EAR, MAR, brow stress, head pose, gaze
│   ├── audio/
│   │   ├── transcriber.py        # faster-whisper wrapper
│   │   └── audio_features.py     # Pitch, WPM, energy, fillers, hedges
│   ├── features/
│   │   ├── fusion.py             # Aligns facial + audio into FusedWindow
│   │   ├── baseline.py           # Personal baseline establishment
│   │   ├── classifier.py         # Rule-based 6-state classifier
│   │   └── session_stats.py      # Running aggregates + delta tracking
│   ├── agent/
│   │   ├── nodes.py              # LangGraph nodes: analyze, nudge, report
│   │   ├── graph.py              # WINDOW_GRAPH + REPORT_GRAPH definitions
│   │   └── runner.py             # Clean public interface
│   ├── output/
│   │   └── session_logger.py     # Persists sessions to HF Dataset or disk
│   └── contracts.py              # Pydantic schemas (FusedWindow, etc.)
├── models/                       # Auto-downloaded at first launch
├── requirements.txt
└── packages.txt                  # Linux system deps for HF Spaces
```

---

## Running Locally

**Prerequisites:** Python 3.9+, Chrome or Edge (for Web Speech API)

```bash
git clone https://github.com/<your-username>/mindmirror.git
cd mindmirror

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env — add your GEMINI_API_KEY
# Or set LLM_PROVIDER=ollama and run: ollama pull llama3.2

python app.py
# Open http://localhost:7860
```

The MediaPipe model (~30 MB) downloads automatically on first launch.

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | `gemini`, `hf`, or `ollama` |
| `GEMINI_API_KEY` | — | Required if `LLM_PROVIDER=gemini` |
| `HF_TOKEN` | — | Required for HF Dataset session storage |
| `HF_DATASET_REPO` | — | e.g. `your-username/mindmirror-sessions` |

---

## HF Spaces Deployment

1. Create a new Space (Gradio SDK)
2. Add the secrets above in **Settings → Repository secrets**
3. Create a **private** HF Dataset repo for session logs
4. Push this repo to the Space remote:

```bash
git remote add hf https://huggingface.co/spaces/<username>/mindmirror
git push hf main
```

System dependencies (`packages.txt`) and model auto-download are already configured.

---

## How Scoring Works

```
Delivery Score (0–10)
  ├── Eye contact percentage
  ├── Confident state proportion
  ├── Filler word frequency
  └── Speaking rate deviation from ideal (120–160 wpm)

Content Score (0–10)
  └── LLM evaluates: did the answer address the question asked?

Final Score = (Delivery / 10) × (Content / 10) × 10
```

This multiplicative formula means technical polish cannot compensate for an irrelevant answer — both dimensions must be strong.

---

## Behavioral States

| State | Trigger conditions |
|-------|--------------------|
| `CONFIDENT` | Strong eye contact + low brow stress + stable head + fluent speech |
| `NERVOUS` | High EAR or low EAR + elevated brow stress + fast speech + fillers |
| `UNCERTAIN` | High hedge density + rising pitch + reduced eye contact |
| `THINKING` | Low speech rate + upward gaze + neutral face |
| `DISENGAGED` | Low energy + minimal movement + poor eye contact |
| `NEUTRAL` | No dominant signal |

All thresholds are relative to the personal baseline, not global norms.

---

## License

MIT © 2025
