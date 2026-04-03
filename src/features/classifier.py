"""
Phase 3: Behavioral state classification.

Rule-based classifier that maps a FusedWindow + baseline
into one of five behavioral states.

Why rule-based instead of ML:
1. No training data available at hackathon time
2. Rules are transparent and explainable to judges
3. Rules can be calibrated in real time based on observation
4. Each rule produces an evidence dict showing exactly why
   the state was assigned — critical for the Gemini prompt

States: CONFIDENT, NERVOUS, UNCERTAIN, THINKING, DISENGAGED, NEUTRAL
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import logging
import time
from typing import Tuple
from src.contracts import FusedWindow, SessionBaseline

logger = logging.getLogger(__name__)


def classify_behavioral_state(
    window: FusedWindow,
    baseline: SessionBaseline
) -> Tuple[str, float, dict]:
    """
    Classify behavioral state from a fused window.

    Rules fire in priority order. First rule exceeding
    its confidence threshold wins.

    Args:
        window:   FusedWindow from fusion.py
        baseline: SessionBaseline from baseline.py

    Returns:
        (state, confidence, evidence)
        state:      str — CONFIDENT/NERVOUS/UNCERTAIN/
                          THINKING/DISENGAGED/NEUTRAL
        confidence: float 0-1 — how strongly state was detected
        evidence:   dict — which specific metrics triggered this,
                           used verbatim in the Gemini prompt
    """
    f = window['facial']
    a = window['audio']
    b = baseline

    # Pre-compute relative metrics
    # (current value relative to personal baseline)
    ear_ratio     = f['avg_ear']            / max(b['avg_ear'], 0.01)
    rate_ratio    = a['speaking_rate_wpm']  / max(b['speaking_rate_wpm'], 1.0)
    pitch_ratio   = a['pitch_variance']     / max(b['pitch_variance'], 1.0)
    energy_ratio  = a['mean_energy']        / max(b['mean_energy'], 0.001)

    # ── NERVOUS ──────────────────────────────────────────────────────────────
    # Multiple stress signals firing together
    nervous_signals = []
    nervous_score   = 0.0

    if ear_ratio < 0.7:
        nervous_signals.append(
            f"eye_openness {f['avg_ear']:.2f} vs baseline {b['avg_ear']:.2f}"
        )
        nervous_score += 0.25

    if rate_ratio > 1.25:
        nervous_signals.append(
            f"speaking_rate {a['speaking_rate_wpm']:.0f}wpm "
            f"vs baseline {b['speaking_rate_wpm']:.0f}wpm"
        )
        nervous_score += 0.25

    if a['filler_count'] >= 2:
        nervous_signals.append(
            f"filler_words {a['filler_count']} "
            f"({', '.join(a['fillers_detected'][:3])})"
        )
        nervous_score += 0.25

    if f['brow_stress'] > b['brow_stress'] * 1.5:
        nervous_signals.append(
            f"brow_stress {f['brow_stress']:.2f} "
            f"vs baseline {b['brow_stress']:.2f}"
        )
        nervous_score += 0.15

    if a['mid_phrase_pauses'] >= 2:
        nervous_signals.append(
            f"mid_phrase_pauses {a['mid_phrase_pauses']}"
        )
        nervous_score += 0.10

    # Require at least 2 signals for NERVOUS
    if nervous_score >= 0.4 and len(nervous_signals) >= 2:
        return (
            "NERVOUS",
            min(nervous_score, 1.0),
            {
                "state":   "NERVOUS",
                "signals": nervous_signals,
                "metrics": {
                    "eye_openness":    round(f['avg_ear'], 3),
                    "speaking_rate":   round(a['speaking_rate_wpm'], 0),
                    "filler_count":    a['filler_count'],
                    "brow_stress":     round(f['brow_stress'], 3),
                    "mid_pauses":      a['mid_phrase_pauses']
                }
            }
        )

    # ── UNCERTAIN ─────────────────────────────────────────────────────────────
    # Hedging language + rising intonation + reduced eye contact
    uncertain_signals = []
    uncertain_score   = 0.0

    if a['hedge_count'] >= 2:
        uncertain_signals.append(
            f"hedges {a['hedge_count']} "
            f"({', '.join(a['hedges_detected'][:3])})"
        )
        uncertain_score += 0.35

    if a['uptalk']:
        uncertain_signals.append("uptalk detected (rising intonation)")
        uncertain_score += 0.25

    if ear_ratio < 0.8:
        uncertain_signals.append(
            f"reduced_eye_contact {f['avg_ear']:.2f} "
            f"vs baseline {b['avg_ear']:.2f}"
        )
        uncertain_score += 0.20

    if a['filler_count'] >= 1:
        uncertain_signals.append(
            f"filler_words {a['filler_count']}"
        )
        uncertain_score += 0.10

    if rate_ratio > 1.15:
        uncertain_signals.append(
            f"elevated_rate {a['speaking_rate_wpm']:.0f}wpm"
        )
        uncertain_score += 0.10

    if uncertain_score >= 0.45 and len(uncertain_signals) >= 2:
        return (
            "UNCERTAIN",
            min(uncertain_score, 1.0),
            {
                "state":   "UNCERTAIN",
                "signals": uncertain_signals,
                "metrics": {
                    "hedge_count":   a['hedge_count'],
                    "hedges":        a['hedges_detected'],
                    "uptalk":        a['uptalk'],
                    "eye_openness":  round(f['avg_ear'], 3),
                    "filler_count":  a['filler_count']
                }
            }
        )

    # ── THINKING ──────────────────────────────────────────────────────────────
    # Deliberate boundary pause + reduced rate + no other stress signals
    # This is a POSITIVE signal — do not interrupt
    thinking_signals = []
    thinking_score   = 0.0

    if a['boundary_pauses'] >= 1 and a['mid_phrase_pauses'] == 0:
        thinking_signals.append(
            f"boundary_pause {a['boundary_pauses']} "
            f"(deliberate pacing)"
        )
        thinking_score += 0.40

    if rate_ratio < 0.75 and a['speaking_rate_wpm'] > 0:
        thinking_signals.append(
            f"reduced_rate {a['speaking_rate_wpm']:.0f}wpm "
            f"vs baseline {b['speaking_rate_wpm']:.0f}wpm"
        )
        thinking_score += 0.30

    # Only THINKING if no stress signals present
    no_stress = (
        a['filler_count'] == 0
        and a['mid_phrase_pauses'] == 0
        and f['brow_stress'] < b['brow_stress'] * 1.3
    )

    if thinking_score >= 0.40 and no_stress:
        return (
            "THINKING",
            min(thinking_score, 1.0),
            {
                "state":   "THINKING",
                "signals": thinking_signals,
                "metrics": {
                    "boundary_pauses": a['boundary_pauses'],
                    "speaking_rate":   round(a['speaking_rate_wpm'], 0),
                    "no_stress":       True
                }
            }
        )

    # ── DISENGAGED ────────────────────────────────────────────────────────────
    # Low energy + monotone + trailing off
    disengaged_signals = []
    disengaged_score   = 0.0

    if energy_ratio < 0.5:
        disengaged_signals.append(
            f"low_energy {a['mean_energy']:.4f} "
            f"vs baseline {b['mean_energy']:.4f}"
        )
        disengaged_score += 0.35

    if pitch_ratio < 0.4:
        disengaged_signals.append(
            f"monotone pitch_variance {a['pitch_variance']:.1f} "
            f"vs baseline {b['pitch_variance']:.1f}"
        )
        disengaged_score += 0.35

    if a['trailing_off']:
        disengaged_signals.append("trailing_off detected")
        disengaged_score += 0.20

    if not f['eye_contact']:
        disengaged_signals.append("no_eye_contact")
        disengaged_score += 0.10

    if disengaged_score >= 0.55 and len(disengaged_signals) >= 2:
        return (
            "DISENGAGED",
            min(disengaged_score, 1.0),
            {
                "state":   "DISENGAGED",
                "signals": disengaged_signals,
                "metrics": {
                    "mean_energy":    round(a['mean_energy'], 4),
                    "pitch_variance": round(a['pitch_variance'], 1),
                    "trailing_off":   a['trailing_off'],
                    "eye_contact":    f['eye_contact']
                }
            }
        )

    # ── CONFIDENT ─────────────────────────────────────────────────────────────
    # All positive signals firing together
    confident_signals = []
    confident_score   = 0.0

    if ear_ratio >= 0.85:
        confident_signals.append(
            f"good_eye_openness {f['avg_ear']:.2f}"
        )
        confident_score += 0.20

    if f['eye_contact']:
        confident_signals.append("eye_contact maintained")
        confident_score += 0.20

    if 0.8 <= rate_ratio <= 1.2 and a['speaking_rate_wpm'] > 0:
        confident_signals.append(
            f"good_speaking_rate {a['speaking_rate_wpm']:.0f}wpm"
        )
        confident_score += 0.20

    if a['filler_count'] == 0:
        confident_signals.append("no_filler_words")
        confident_score += 0.15

    if a['hedge_count'] <= 1:
        confident_signals.append("minimal_hedging")
        confident_score += 0.10

    if pitch_ratio >= 0.7 and a['pitch_variance'] > 0:
        confident_signals.append(
            f"expressive_pitch variance={a['pitch_variance']:.1f}"
        )
        confident_score += 0.15

    if confident_score >= 0.55 and len(confident_signals) >= 3:
        return (
            "CONFIDENT",
            min(confident_score, 1.0),
            {
                "state":   "CONFIDENT",
                "signals": confident_signals,
                "metrics": {
                    "eye_openness":   round(f['avg_ear'], 3),
                    "eye_contact":    f['eye_contact'],
                    "speaking_rate":  round(a['speaking_rate_wpm'], 0),
                    "filler_count":   a['filler_count'],
                    "hedge_count":    a['hedge_count'],
                    "pitch_variance": round(a['pitch_variance'], 1)
                }
            }
        )

    # ── NEUTRAL ───────────────────────────────────────────────────────────────
    # Nothing fired strongly enough
    return (
        "NEUTRAL",
        0.5,
        {
            "state":   "NEUTRAL",
            "signals": ["no strong signal detected"],
            "metrics": {
                "ear_ratio":    round(ear_ratio, 2),
                "rate_ratio":   round(rate_ratio, 2),
                "pitch_ratio":  round(pitch_ratio, 2),
                "energy_ratio": round(energy_ratio, 2)
            }
        }
    )
