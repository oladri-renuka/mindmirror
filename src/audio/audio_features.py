"""
Phase 2: Audio feature computation from raw audio + transcript.

Uses yin for pitch (fastest accurate method on M4).
All features return interpretable values with clear units.

Feature computation is separated from transcription so
they can run on different threads if needed later.
"""

# Prevent OpenMP segfault on Apple Silicon M4
# Must be set before librosa import
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import librosa
import time
import logging
import re
from typing import List, Tuple, Optional
from src.contracts import AudioFeatureVector

logger = logging.getLogger(__name__)

SR = 16000  # Expected sample rate for all input audio


# ── Filler and hedge word lists ──────────────────────────────────────────────

FILLER_WORDS = [
    r'\bum\b', r'\buh\b', r'\ber\b',
    r'\blike\b', r'\byou know\b', r'\bbasically\b',
    r'\bliterally\b', r'\bsort of\b', r'\bkind of\b',
    r'\bright\b', r'\bso\b', r'\bactually\b'
]

HEDGE_PHRASES = [
    r'\bi think\b', r'\bi believe\b', r'\bmaybe\b',
    r'\bprobably\b', r"\bi'm not sure\b", r'\bi guess\b',
    r'\bi feel like\b', r'\bit seems\b', r'\bperhaps\b',
    r'\bif i recall\b', r'\bsomething like\b'
]


# ── Individual feature functions ─────────────────────────────────────────────

def compute_pitch_features(
    audio: np.ndarray,
    sr: int = SR
) -> dict:
    """
    Compute pitch features using yin algorithm.

    yin is ~100x faster than pyin with comparable accuracy
    for speech pitch range (80-400 Hz).

    Returns:
        mean_pitch: float, mean F0 in Hz (0 if unvoiced)
        pitch_variance: float, std dev of F0 (higher = more expressive)
        uptalk: bool, True if pitch rises at end of chunk
                (rising intonation = uncertainty signal)
    """
    try:
        f0 = librosa.yin(
            audio,
            fmin=80,
            fmax=400,
            sr=sr
        )

        # Filter out near-zero values (silence/unvoiced)
        voiced = f0[f0 > 50]

        if len(voiced) == 0:
            return {
                "mean_pitch":     0.0,
                "pitch_variance": 0.0,
                "uptalk":         False
            }

        mean_pitch     = float(np.mean(voiced))
        pitch_variance = float(np.std(voiced))

        # Uptalk: compare mean pitch of last 20% vs first 80%
        # Rising pitch at end of a statement = uncertainty
        split_idx = int(len(f0) * 0.8)
        first_part  = f0[:split_idx]
        last_part   = f0[split_idx:]

        first_voiced = first_part[first_part > 50]
        last_voiced  = last_part[last_part  > 50]

        if len(first_voiced) > 3 and len(last_voiced) > 3:
            uptalk = float(np.mean(last_voiced)) > float(np.mean(first_voiced)) * 1.1
        else:
            uptalk = False

        return {
            "mean_pitch":     mean_pitch,
            "pitch_variance": pitch_variance,
            "uptalk":         bool(uptalk)
        }

    except Exception as e:
        logger.warning(f"Pitch computation failed: {e}")
        return {
            "mean_pitch":     0.0,
            "pitch_variance": 0.0,
            "uptalk":         False
        }


def compute_speaking_rate(
    audio: np.ndarray,
    transcript_words: list,
    sr: int = SR
) -> float:
    """
    Compute speaking rate in words per minute.

    Uses word count from Whisper transcript divided by
    actual audio duration. Falls back to syllable-rate
    proxy if transcript is empty.

    Returns:
        Speaking rate in WPM (0 if silence)
    """
    duration_seconds = len(audio) / sr
    if duration_seconds == 0:
        return 0.0

    # Primary: use Whisper word count
    word_count = len([
        w for w in transcript_words
        if w.get("word", "").strip()
    ])

    if word_count > 0:
        wpm = (word_count / duration_seconds) * 60.0
        return float(wpm)

    # No speech detected — return 0
    return 0.0


def compute_energy_features(
    audio: np.ndarray,
    sr: int = SR
) -> dict:
    """
    Compute energy/volume features.

    Returns:
        mean_energy: float, overall volume level
        energy_variance: float, expressiveness measure
                         (monotone delivery = low variance)
        trailing_off: bool, True if volume drops in second half
                      (indicates losing confidence mid-answer)
    """
    try:
        rms = librosa.feature.rms(y=audio)[0]

        mean_energy     = float(np.mean(rms))
        energy_variance = float(np.var(rms))

        # Trailing off: second half quieter than first half
        mid = len(rms) // 2
        first_half_energy  = float(np.mean(rms[:mid]))
        second_half_energy = float(np.mean(rms[mid:]))

        trailing_off = (
            second_half_energy < first_half_energy * 0.7
            and mean_energy > 0.01  # Not just silence
        )

        return {
            "mean_energy":     mean_energy,
            "energy_variance": energy_variance,
            "trailing_off":    bool(trailing_off)
        }

    except Exception as e:
        logger.warning(f"Energy computation failed: {e}")
        return {
            "mean_energy":     0.0,
            "energy_variance": 0.0,
            "trailing_off":    False
        }


def compute_pause_features(
    audio: np.ndarray,
    transcript_text: str,
    sr: int = SR,
    min_silence_duration: float = 0.4
) -> dict:
    """
    Detect pauses and classify them as boundary or mid-phrase.

    Boundary pause: comes after sentence-ending punctuation
                    → deliberate, shows pacing (neutral/good)
    Mid-phrase pause: comes mid-sentence
                    → losing thread, uncertainty (negative)

    Returns:
        boundary_pauses: int
        mid_phrase_pauses: int
    """
    try:
        # Find silent regions
        intervals = librosa.effects.split(
            audio,
            top_db=35,      # Lower = more sensitive to silence
            frame_length=512,
            hop_length=128
        )

        # Convert to silence regions (gaps between speech intervals)
        silence_regions = []
        for i in range(len(intervals) - 1):
            silence_start = intervals[i][1] / sr
            silence_end   = intervals[i + 1][0] / sr
            duration      = silence_end - silence_start

            if duration >= min_silence_duration:
                silence_regions.append({
                    "start":    silence_start,
                    "end":      silence_end,
                    "duration": duration
                })

        if not silence_regions:
            return {"boundary_pauses": 0, "mid_phrase_pauses": 0}

        # Classify each pause using transcript
        # Simple heuristic: if transcript has punctuation,
        # estimate position proportionally
        boundary_pauses   = 0
        mid_phrase_pauses = 0

        total_duration = len(audio) / sr

        # Find punctuation positions as fraction of total duration
        boundary_markers = []
        if transcript_text:
            words = transcript_text.split()
            for i, word in enumerate(words):
                if word.endswith(('.', ',', '?', '!', ';', ':')):
                    # Approximate time position of this word
                    pos = (i / max(len(words), 1)) * total_duration
                    boundary_markers.append(pos)

        for silence in silence_regions:
            silence_midpoint = (silence["start"] + silence["end"]) / 2

            # Is this pause near a boundary marker?
            near_boundary = any(
                abs(silence_midpoint - bm) < 0.5
                for bm in boundary_markers
            )

            if near_boundary or not boundary_markers:
                boundary_pauses += 1
            else:
                mid_phrase_pauses += 1

        return {
            "boundary_pauses":   boundary_pauses,
            "mid_phrase_pauses": mid_phrase_pauses
        }

    except Exception as e:
        logger.warning(f"Pause computation failed: {e}")
        return {"boundary_pauses": 0, "mid_phrase_pauses": 0}


def detect_filler_words(transcript_text: str) -> dict:
    """
    Detect filler words via regex pattern matching.

    Deliberately simple — fillers are syntactically distinctive
    and don't need semantic understanding.

    Returns:
        filler_count: int
        fillers_detected: list of str
    """
    if not transcript_text:
        return {"filler_count": 0, "fillers_detected": []}

    text_lower = transcript_text.lower()
    detected = []

    for pattern in FILLER_WORDS:
        matches = re.findall(pattern, text_lower)
        detected.extend(matches)

    return {
        "filler_count":     len(detected),
        "fillers_detected": detected
    }


def detect_hedge_phrases(transcript_text: str) -> dict:
    """
    Detect hedging language via regex pattern matching.

    Hedges signal uncertainty: "I think", "maybe", "probably"
    Frequent hedging combined with other signals = UNCERTAIN state.

    Returns:
        hedge_count: int
        hedges_detected: list of str
    """
    if not transcript_text:
        return {"hedge_count": 0, "hedges_detected": []}

    text_lower = transcript_text.lower()
    detected = []

    for pattern in HEDGE_PHRASES:
        matches = re.findall(pattern, text_lower)
        detected.extend(matches)

    return {
        "hedge_count":     len(detected),
        "hedges_detected": detected
    }


# ── Master function ───────────────────────────────────────────────────────────

def extract_audio_feature_vector(
    audio_chunk: np.ndarray,
    transcription_result: dict,
    sr: int = SR
) -> AudioFeatureVector:
    """
    Master function: compute all audio features from one chunk.

    This is the only function other modules should call.
    Takes pre-computed transcription result to avoid
    running Whisper twice.

    Args:
        audio_chunk: float32 numpy array at 16000Hz, shape (32000,)
        transcription_result: dict from Transcriber.transcribe()
        sr: sample rate, default 16000

    Returns:
        AudioFeatureVector with all features + timestamp
    """
    transcript_text  = transcription_result.get("text",  "")
    transcript_words = transcription_result.get("words", [])

    # Run all feature extractors
    pitch    = compute_pitch_features(audio_chunk, sr)
    energy   = compute_energy_features(audio_chunk, sr)
    pauses   = compute_pause_features(audio_chunk, transcript_text, sr)
    fillers  = detect_filler_words(transcript_text)
    hedges   = detect_hedge_phrases(transcript_text)
    rate     = compute_speaking_rate(audio_chunk, transcript_words, sr)

    return AudioFeatureVector(
        # Pitch
        mean_pitch=pitch["mean_pitch"],
        pitch_variance=pitch["pitch_variance"],
        uptalk=pitch["uptalk"],

        # Rate
        speaking_rate_wpm=rate,

        # Energy
        mean_energy=energy["mean_energy"],
        energy_variance=energy["energy_variance"],
        trailing_off=energy["trailing_off"],

        # Pauses
        boundary_pauses=pauses["boundary_pauses"],
        mid_phrase_pauses=pauses["mid_phrase_pauses"],

        # Language
        filler_count=fillers["filler_count"],
        fillers_detected=fillers["fillers_detected"],
        hedge_count=hedges["hedge_count"],
        hedges_detected=hedges["hedges_detected"],

        # Transcript
        transcript=transcript_text,

        # Metadata
        timestamp=time.time()
    )
