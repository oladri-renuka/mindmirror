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

**Practice interview answers and get real-time feedback on whether you *look* confident — not just whether your words sound right.**

MindMirror analyzes your face and voice every 2 seconds, detects behavioral signals like eye contact, brow stress, filler words, and speaking pace, then fires live AI coaching nudges and generates a scored report at the end.

</div>

---

## What it detects

| Signal | Method |
|--------|--------|
| **Eye contact** | MediaPipe 478-point face mesh → gaze vector projection |
| **Facial state** | Eye Aspect Ratio, Mouth Aspect Ratio, brow stress, head pose (yaw/pitch) |
| **Speech** | faster-whisper (base, int8) — real-time transcription |
| **Vocal patterns** | Pitch (YIN algorithm), speaking rate (WPM), pause detection |
| **Filler/hedge words** | Token-level: "um", "uh", "like", "you know", "kind of", etc. |

These signals are fused every 2 seconds and classified into one of six **behavioral states**, each calibrated against your personal baseline from the first 15 seconds:

| State | What triggers it |
|-------|-----------------|
| `CONFIDENT` | Strong eye contact + low brow stress + stable head + fluent speech |
| `NERVOUS` | Elevated brow stress + fast speech + high filler frequency |
| `UNCERTAIN` | High hedge density + rising pitch + reduced eye contact |
| `THINKING` | Low speech rate + upward gaze + neutral face |
| `DISENGAGED` | Low energy + minimal movement + poor eye contact |
| `NEUTRAL` | No dominant signal |

> All thresholds are relative to *your* resting baseline — not a global average. A naturally fast speaker won't be flagged as nervous just for speaking quickly.

---

## How scoring works

```
Delivery Score (0–10)
  ├── Eye contact percentage
  ├── Confident state proportion
  ├── Filler word frequency
  └── Speaking rate deviation from ideal (120–160 wpm)

Content Score (0–10)
  └── LLM evaluates: did the answer actually address the question?

Final Score = (Delivery / 10) × (Content / 10) × 10
```

The multiplicative formula means you can't compensate for an off-topic answer with confident delivery — both dimensions have to be strong.

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
        └── faster-whisper ────────▶       │
                                           ▼
                                    PersonalBaseline calibration
                                           │
                                    BehavioralStateClassifier
                                           │
                                    LangGraph Agent ──▶ Nudge / Report
                                           │
                                    SessionLogger (HF Dataset / disk)
```

**Key design decisions:**
- **Browser-mode pipeline** — no server-side mic or webcam threads; Gradio stream callbacks push data in. Runs identically on HF Spaces and local dev.
- **Personal baseline** — first 7 windows (~15s) establish your resting state so the classifier is relative to you, not a global norm.
- **Multiplicative scoring** — delivery polish can't compensate for irrelevant content.
- **LLM fallback chain** — Gemini 2.0 Flash → HF Inference API → rule-based. No single point of failure.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| UI | Gradio 6.9 |
| Vision | MediaPipe 0.10.33, OpenCV |
| Speech | faster-whisper (base, int8) |
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

The 2-second timer fires well within budget.

---

## Project structure

```
mindmirror/
├── app.py                        # Gradio UI
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
├── models/                       # Auto-downloaded at first launch (~30 MB)
├── requirements.txt
└── packages.txt                  # Linux system deps for HF Spaces
```

---

## Setup

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_PROVIDER` | No (default: `gemini`) | `gemini`, `hf`, or `ollama` |
| `GEMINI_API_KEY` | If `LLM_PROVIDER=gemini` | Google Gemini API key |
| `HF_TOKEN` | For session storage | HF token with write access |
| `HF_DATASET_REPO` | For session storage | e.g. `your-username/mindmirror-sessions` |

### Local development

```bash
git clone https://github.com/<your-username>/mindmirror.git
cd mindmirror

python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env   # add GEMINI_API_KEY, or set LLM_PROVIDER=ollama
python app.py          # opens http://localhost:7860
```

The MediaPipe model (~30 MB) downloads automatically on first launch.

### HF Spaces deployment

1. Create a new Space (Gradio SDK)
2. Add the environment variables above under **Settings → Repository secrets**
3. Create a **private** HF Dataset repo for session logs
4. Push:

```bash
git remote add hf https://huggingface.co/spaces/<username>/mindmirror
git push hf main
```

System packages and model auto-download are pre-configured.

---

## License

MIT © 2025
