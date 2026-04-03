"""
Pipeline orchestrator — browser-mode design.

All input arrives via push methods called by Gradio stream callbacks.
No hardware threads, no cv2.VideoCapture, no PyAudio.
Compatible with HF Spaces and local Gradio dev.

Usage:
    pipeline = MindMirrorPipeline()

    # Gradio webcam stream callback:
    pipeline.push_video_frame(frame)          # numpy RGB array

    # Gradio mic stream callback:
    pipeline.push_audio_chunk(rate, data)     # (int, numpy array)

    # Gradio 2-second Timer:
    result = pipeline.process_tick()

    # End Session button:
    report = pipeline.end_session()
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import hashlib
import time
import threading
import logging
import numpy as np
from collections import deque
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Audio constants — Whisper requires 16 kHz
_AUDIO_RATE   = 16_000
_AUDIO_CHUNK  = _AUDIO_RATE * 2       # 2-second window
_AUDIO_BUFFER = _AUDIO_RATE * 6       # 6-second circular buffer

# ── Phase imports ─────────────────────────────────────────────────────────────
from src.vision.face_tracker     import FaceTracker
from src.vision.facial_features  import extract_facial_feature_vector
from src.audio.transcriber       import Transcriber
from src.audio.audio_features    import extract_audio_feature_vector
from src.features.fusion         import create_fused_window
from src.features.baseline       import BaselineEstablisher
from src.features.classifier     import classify_behavioral_state
from src.features.session_stats  import SessionStats
from src.agent.runner            import get_initial_state, process_window, generate_report
from src.output.session_logger   import SessionLogger


class MindMirrorPipeline:
    """
    Manages the complete MindMirror session lifecycle.

    Browser-mode: video frames and audio chunks are pushed in by
    Gradio stream callbacks instead of read from local hardware.
    Works identically on HF Spaces and local Gradio dev server.
    """

    def __init__(self):
        # ── Vision / audio models ─────────────────────────────────────
        self._tracker     = FaceTracker(model_path="models/face_landmarker.task")
        self._transcriber = Transcriber(model_size="tiny")

        # ── Thread-safe input buffers ─────────────────────────────────
        # Facial feature vectors — populated by push_video_frame()
        self._facial_buffer: list = []
        self._facial_lock         = threading.Lock()

        # Audio samples at 16 kHz — circular, populated by push_audio_chunk()
        self._audio_buffer: deque = deque(maxlen=_AUDIO_BUFFER)
        self._audio_lock          = threading.Lock()

        # ── Session state ─────────────────────────────────────────────
        self._baseline_est:    Optional[BaselineEstablisher] = None
        self._session_stats:   Optional[SessionStats]        = None
        self._agent_state:     Optional[dict]                = None
        self._baseline:        Optional[dict]                = None

        self._window_start     = 0.0
        self._last_process     = 0.0
        self._last_audio_hash  = None

        self._session_active   = False
        self._session_start    = 0.0
        self._current_question = ""
        self._username         = "anonymous"

        self._latest_result    = self._empty_result()

        # ── Session recording (for persistence) ───────────────────────
        self._state_timeline:    list = []
        self._nudges_given:      list = []
        self._transcript_chunks: list = []

        logger.info("Pipeline initialised (browser mode)")

    # ── Push methods (called by Gradio stream callbacks) ──────────────────────

    def push_video_frame(self, frame: np.ndarray):
        """
        Accept one browser webcam frame (RGB numpy array from Gradio).
        Runs MediaPipe on it and appends to the facial feature buffer.
        Called ~7-10 fps by the webcam stream event.
        """
        if frame is None:
            return

        # Gradio sends RGB; MediaPipe / OpenCV expect BGR
        frame_bgr  = frame[..., ::-1].copy()
        face_result = self._tracker.process_frame(frame_bgr)

        if face_result is not None:
            fv = extract_facial_feature_vector(face_result)
            with self._facial_lock:
                self._facial_buffer.append(fv)

    def push_audio_chunk(self, sample_rate: int, audio_data: np.ndarray):
        """
        Accept one browser mic chunk.
        Normalises to float32, resamples to 16 kHz, appends to buffer.
        Called frequently by the mic stream event (e.g. every ~100 ms).
        """
        if audio_data is None or audio_data.size == 0:
            return

        # Stereo → mono
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        # Normalise to float32 in [-1, 1]
        if audio_data.dtype != np.float32:
            if np.issubdtype(audio_data.dtype, np.integer):
                audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
            else:
                audio_data = audio_data.astype(np.float32)

        # Resample to 16 kHz using linear interpolation
        if sample_rate and sample_rate != _AUDIO_RATE:
            new_len = int(len(audio_data) * _AUDIO_RATE / sample_rate)
            if new_len > 0:
                old_idx    = np.linspace(0, len(audio_data) - 1, new_len)
                audio_data = np.interp(old_idx, np.arange(len(audio_data)),
                                       audio_data).astype(np.float32)

        with self._audio_lock:
            self._audio_buffer.extend(audio_data.tolist())

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def start_session(
        self,
        question:        str = "Tell me about yourself",
        question_number: int = 1,
        total_questions: int = 5,
        username:        str = "anonymous",
    ):
        """Initialise a fresh session. Clears all buffers."""
        self._baseline_est  = BaselineEstablisher(target_windows=7)
        self._session_stats = SessionStats()
        self._agent_state   = get_initial_state(
            question=question,
            question_number=question_number,
            total_questions=total_questions,
        )
        self._baseline         = None
        self._last_audio_hash  = None
        self._window_start     = time.time()
        self._last_process     = time.time()
        self._session_active   = True
        self._session_start    = time.time()
        self._current_question = question
        self._username         = username
        self._latest_result    = self._empty_result()

        self._state_timeline    = []
        self._nudges_given      = []
        self._transcript_chunks = []

        with self._facial_lock:
            self._facial_buffer = []
        with self._audio_lock:
            self._audio_buffer.clear()

        logger.info(f"Session started: '{question}' for '{username}'")

    def process_tick(self) -> dict:
        """
        Called every 2 seconds by Gradio Timer.
        Reads from internal buffers, runs the full ML pipeline,
        returns a result dict for the UI.
        """
        if not self._session_active:
            return self._latest_result

        now     = time.time()
        elapsed = now - self._session_start

        # Throttle to every 2 seconds
        if now - self._last_process < 2.0:
            return self._latest_result

        # Snapshot and clear facial buffer
        with self._facial_lock:
            facial_snapshot      = self._facial_buffer.copy()
            self._facial_buffer  = []

        if not facial_snapshot:
            return self._latest_result

        # Snapshot last 2 seconds of audio (use silence if mic not ready yet)
        with self._audio_lock:
            buf_len = len(self._audio_buffer)
            if buf_len >= _AUDIO_CHUNK:
                audio_chunk = np.array(
                    list(self._audio_buffer)[-_AUDIO_CHUNK:], dtype=np.float32
                )
            elif buf_len > 0:
                audio_chunk = np.array(list(self._audio_buffer), dtype=np.float32)
            else:
                # No mic data yet — process facial data only with silence
                audio_chunk = np.zeros(_AUDIO_CHUNK // 4, dtype=np.float32)

        self._last_process = now
        window_end         = now

        # Deduplicate — skip if same audio chunk as last tick
        chunk_hash = hashlib.md5(
            audio_chunk[:min(100, len(audio_chunk))].tobytes() +
            audio_chunk[-min(100, len(audio_chunk)):].tobytes()
        ).hexdigest()[:8]
        if chunk_hash == self._last_audio_hash:
            return self._latest_result
        self._last_audio_hash = chunk_hash

        # ── Transcribe ────────────────────────────────────────────────
        transcript     = self._transcriber.transcribe(audio_chunk)
        audio_features = extract_audio_feature_vector(audio_chunk, transcript)

        # ── Fuse ──────────────────────────────────────────────────────
        fused = create_fused_window(
            facial_vectors=facial_snapshot,
            audio_vector=audio_features,
            window_start=self._window_start,
            window_end=window_end,
        )
        self._window_start = window_end

        # ── Baseline calibration ──────────────────────────────────────
        if not self._baseline_est.is_ready():
            self._baseline_est.add_window(fused)
            windows_done = self._baseline_est.windows_collected()

            result = self._empty_result()
            result.update({
                "phase":           "calibrating",
                "elapsed":         elapsed,
                "calibration_pct": windows_done / 7.0,
                "transcript":      transcript["text"],
                "status_text":     f"Calibrating your baseline... ({windows_done}/7 windows)",
            })
            if self._baseline_est.is_ready():
                self._baseline          = self._baseline_est.get_baseline()
                result["status_text"]   = "Baseline ready — coaching active!"
                result["baseline_ready"] = True

            self._latest_result = result
            return result

        if self._baseline is None:
            self._baseline = self._baseline_est.force_baseline()

        # ── Classify ──────────────────────────────────────────────────
        state_label, confidence, evidence = classify_behavioral_state(
            fused, self._baseline
        )

        # ── Session stats ─────────────────────────────────────────────
        delta = self._session_stats.get_delta(state_label)
        self._session_stats.update(state_label, confidence, evidence, fused)
        stats = self._session_stats.get_stats()

        # ── Agent ─────────────────────────────────────────────────────
        nudge_text, is_milestone, self._agent_state = process_window(
            fused_window=fused,
            state_label=state_label,
            confidence=confidence,
            evidence=evidence,
            session_stats=stats,
            baseline=self._baseline,
            agent_state=self._agent_state,
        )

        # ── Record for persistence ────────────────────────────────────
        self._state_timeline.append({"elapsed": elapsed, "state": state_label})
        chunk_text = transcript.get("text", "").strip()
        if chunk_text:
            self._transcript_chunks.append(chunk_text)
        if nudge_text:
            self._nudges_given.append({
                "elapsed":      elapsed,
                "text":         nudge_text,
                "is_milestone": is_milestone,
            })

        # ── Build result for Gradio ───────────────────────────────────
        facial = fused["facial"]
        audio  = fused["audio"]

        result = {
            "phase":              "coaching",
            "elapsed":            elapsed,
            "calibration_pct":    1.0,
            "baseline_ready":     True,
            "state":              state_label,
            "confidence":         confidence,
            "delta":              delta,
            "eye_contact":        facial.get("eye_contact",     False),
            "eye_contact_pct":    stats.get("eye_contact_pct",  0),
            "brow_stress":        facial.get("brow_stress",     0),
            "head_yaw":           facial.get("head_yaw",        0),
            "transcript":         transcript["text"],
            "speaking_rate":      audio.get("speaking_rate_wpm", 0),
            "mean_pitch":         audio.get("mean_pitch",        0),
            "filler_count":       audio.get("filler_count",      0),
            "hedge_count":        audio.get("hedge_count",       0),
            "fillers":            audio.get("fillers_detected",  []),
            "hedges":             audio.get("hedges_detected",   []),
            "total_fillers":      stats.get("total_fillers",     0),
            "total_hedges":       stats.get("total_hedges",      0),
            "window_count":       stats.get("window_count",      0),
            "dominant_state":     stats.get("dominant_state",    "NEUTRAL"),
            "state_distribution": stats.get("state_distribution", {}),
            "weak_patterns":      stats.get("weak_patterns",     []),
            "nudge_text":         nudge_text,
            "is_milestone":       is_milestone,
            "status_text":        f"Coaching active — {state_label}",
        }

        self._latest_result = result
        return result

    def end_session(self) -> str:
        """End session, generate report, persist to storage."""
        if not self._session_active:
            return "No active session."

        self._session_active = False
        logger.info("Session ended — generating report")

        if self._agent_state is None:
            return (
                "⚠️ Session too short — no data was collected.\n\n"
                "Make sure your webcam and microphone are active before clicking Start Session. "
                "Allow at least 15 seconds for baseline calibration before ending."
            )

        window_count = len(self._agent_state.get("state_history", []))
        if window_count < 3:
            return (
                f"⚠️ Session too short to generate a meaningful report ({window_count} window(s) recorded).\n\n"
                "Tips:\n"
                "• Activate the microphone widget before starting\n"
                "• Allow at least 30 seconds before ending the session\n"
                "• Make sure your face is visible in the webcam feed"
            )

        # Inject full transcript for content scoring
        self._agent_state["full_transcript"] = " ".join(self._transcript_chunks)

        result = generate_report(self._agent_state)
        report = result["report"]

        duration = time.time() - self._session_start
        stats    = self._session_stats.get_stats() if self._session_stats else {}
        backend  = "hf" if os.environ.get("SPACE_ID") else "local"

        try:
            SessionLogger(username=self._username, backend=backend).save(
                question          = self._current_question,
                duration_seconds  = duration,
                state_timeline    = self._state_timeline,
                state_distribution= stats.get("state_distribution", {}),
                metrics           = {
                    "total_fillers":   stats.get("total_fillers",   0),
                    "total_hedges":    stats.get("total_hedges",    0),
                    "eye_contact_pct": stats.get("eye_contact_pct", 0.0),
                    "window_count":    stats.get("window_count",    0),
                },
                nudges            = self._nudges_given,
                report            = report,
                delivery_score    = result.get("delivery_score"),
                content_score     = result.get("content_score"),
                final_score       = result.get("final_score"),
            )
        except Exception as e:
            logger.warning(f"Failed to save session: {e}")

        return report

    def is_active(self) -> bool:
        return self._session_active

    def get_elapsed(self) -> float:
        return time.time() - self._session_start if self._session_active else 0.0

    def shutdown(self):
        """Release all resources."""
        self._session_active = False
        self._tracker.close()
        logger.info("Pipeline shutdown complete")

    def _empty_result(self) -> dict:
        return {
            "phase":              "idle",
            "elapsed":            0.0,
            "calibration_pct":    0.0,
            "baseline_ready":     False,
            "state":              "NEUTRAL",
            "confidence":         0.0,
            "delta":              "STABLE",
            "eye_contact":        False,
            "eye_contact_pct":    0.0,
            "brow_stress":        0.0,
            "head_yaw":           0.0,
            "transcript":         "",
            "speaking_rate":      0.0,
            "mean_pitch":         0.0,
            "filler_count":       0,
            "hedge_count":        0,
            "fillers":            [],
            "hedges":             [],
            "total_fillers":      0,
            "total_hedges":       0,
            "window_count":       0,
            "dominant_state":     "NEUTRAL",
            "state_distribution": {},
            "weak_patterns":      [],
            "nudge_text":         None,
            "is_milestone":       False,
            "status_text":        "Ready to start",
        }
