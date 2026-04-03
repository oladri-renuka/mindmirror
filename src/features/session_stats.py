"""
Phase 3: Session statistics and delta tracking.

Maintains running aggregates across all windows in the session.
Computes delta (IMPROVING/REGRESSING/STABLE) from state history.
Identifies weak patterns for the agent to reference.

These statistics are passed to the LangGraph agent every 2 seconds
so it has full context about the arc of the session, not just
the current window.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import logging
import time
from typing import List, Optional
from collections import Counter

logger = logging.getLogger(__name__)

# State ordering for delta computation
# Higher index = more positive state
STATE_SCORE = {
    "DISENGAGED": 0,
    "NERVOUS":    1,
    "UNCERTAIN":  2,
    "NEUTRAL":    3,
    "THINKING":   4,
    "CONFIDENT":  5
}


def compute_delta(
    current_state: str,
    state_history: List[str]
) -> str:
    """
    Compute trajectory direction from state history.

    Compares current state score against average of last 3 states.

    Returns:
        "IMPROVING"  — trending positive
        "REGRESSING" — trending negative
        "STABLE"     — no significant change
    """
    if len(state_history) < 2:
        return "STABLE"

    current_score = STATE_SCORE.get(current_state, 3)

    # Average score of last 3 states (or fewer if not enough history)
    recent = state_history[-3:]
    avg_recent = sum(
        STATE_SCORE.get(s, 3) for s in recent
    ) / len(recent)

    diff = current_score - avg_recent

    if diff >= 1.0:
        return "IMPROVING"
    elif diff <= -1.0:
        return "REGRESSING"
    else:
        return "STABLE"


def identify_weak_patterns(
    state_history: List[str],
    evidence_history: List[dict],
    threshold_pct: float = 0.30
) -> List[str]:
    """
    Identify recurring behavioral patterns worth flagging.

    A pattern is "weak" if it appears in more than threshold_pct
    of all windows so far.

    Args:
        state_history:    All state labels so far
        evidence_history: All evidence dicts so far
        threshold_pct:    Fraction of windows needed to flag

    Returns:
        List of pattern description strings
    """
    if not state_history:
        return []

    total = len(state_history)
    patterns = []

    # State frequency patterns
    state_counts = Counter(state_history)

    if state_counts.get("NERVOUS", 0) / total > threshold_pct:
        patterns.append(
            f"nervous in {state_counts['NERVOUS']}/{total} windows"
        )

    if state_counts.get("UNCERTAIN", 0) / total > threshold_pct:
        patterns.append(
            f"uncertain in {state_counts['UNCERTAIN']}/{total} windows"
        )

    if state_counts.get("DISENGAGED", 0) / total > threshold_pct:
        patterns.append(
            f"disengaged in {state_counts['DISENGAGED']}/{total} windows"
        )

    # Filler word patterns
    total_fillers = sum(
        e.get('metrics', {}).get('filler_count', 0)
        for e in evidence_history
    )
    if total > 0 and total_fillers / total > 1.5:
        patterns.append(
            f"avg {total_fillers/total:.1f} fillers per window"
        )

    # Eye contact patterns
    ec_values = [
        e.get('metrics', {}).get('eye_contact', True)
        for e in evidence_history
        if 'eye_contact' in e.get('metrics', {})
    ]
    if ec_values:
        ec_false_pct = sum(
            1 for ec in ec_values if not ec
        ) / len(ec_values)
        if ec_false_pct > threshold_pct:
            patterns.append(
                f"eye contact breaks in {ec_false_pct:.0%} of windows"
            )

    return patterns


class SessionStats:
    """
    Maintains running session statistics across all windows.

    Updated after every window. Passed to agent for context.

    Usage:
        stats = SessionStats()

        # After each window:
        stats.update(state, confidence, evidence, fused_window)

        # Get current stats dict for agent:
        current = stats.get_stats()
    """

    def __init__(self):
        self._state_history:    List[str]  = []
        self._evidence_history: List[dict] = []
        self._window_count      = 0
        self._session_start     = time.time()

        # Running totals
        self._total_eye_contact_frames = 0
        self._total_frames             = 0
        self._total_fillers            = 0
        self._total_hedges             = 0
        self._speaking_rates           = []
        self._pitch_values             = []

    def update(
        self,
        state: str,
        confidence: float,
        evidence: dict,
        fused_window
    ):
        """
        Update stats after a new window is processed.

        Args:
            state:        Current behavioral state label
            confidence:   Classifier confidence 0-1
            evidence:     Evidence dict from classifier
            fused_window: The FusedWindow that was classified
        """
        self._window_count += 1
        self._state_history.append(state)
        self._evidence_history.append(evidence)

        f = fused_window['facial']
        a = fused_window['audio']

        # Eye contact
        if f.get('eye_contact'):
            self._total_eye_contact_frames += 1
        self._total_frames += 1

        # Language
        self._total_fillers += a.get('filler_count', 0)
        self._total_hedges  += a.get('hedge_count',  0)

        # Speaking rate — include all windows with speech
        # Use 20wpm minimum to filter pure silence
        rate = a.get('speaking_rate_wpm', 0)
        if rate >= 20:
            self._speaking_rates.append(rate)

        # Pitch
        pitch = a.get('mean_pitch', 0)
        if pitch > 0:
            self._pitch_values.append(pitch)

    def get_delta(self, current_state: str) -> str:
        """Get delta for current state vs recent history."""
        return compute_delta(current_state, self._state_history)

    def get_weak_patterns(self) -> List[str]:
        """Get list of recurring weak patterns."""
        return identify_weak_patterns(
            self._state_history,
            self._evidence_history
        )

    def get_stats(self) -> dict:
        """
        Get complete session statistics dictionary.
        This is what gets passed to the LangGraph agent.
        """
        elapsed = time.time() - self._session_start

        ec_pct = (
            self._total_eye_contact_frames / self._total_frames
            if self._total_frames > 0 else 0.0
        )

        avg_rate = (
            sum(self._speaking_rates) / len(self._speaking_rates)
            if self._speaking_rates else 0.0
        )

        # State distribution
        state_dist = Counter(self._state_history)

        return {
            "window_count":          self._window_count,
            "session_elapsed":       round(elapsed, 0),
            "eye_contact_pct":       round(ec_pct, 3),
            "total_fillers":         self._total_fillers,
            "total_hedges":          self._total_hedges,
            "avg_speaking_rate":     round(avg_rate, 0),
            "state_distribution":    dict(state_dist),
            "state_history":         self._state_history.copy(),
            "weak_patterns":         self.get_weak_patterns(),
            "dominant_state":        (
                state_dist.most_common(1)[0][0]
                if state_dist else "NEUTRAL"
            )
        }

    def get_top_moments(self, n: int = 3) -> List[dict]:
        """
        Get the n most significant moments from the session.

        Significance = largest delta change between consecutive states.
        Used in the final session report.
        """
        if len(self._state_history) < 2:
            return []

        moments = []
        for i in range(1, len(self._state_history)):
            prev  = self._state_history[i - 1]
            curr  = self._state_history[i]
            delta = abs(
                STATE_SCORE.get(curr,  3) -
                STATE_SCORE.get(prev, 3)
            )
            if delta >= 2:
                moments.append({
                    "window_index": i,
                    "from_state":   prev,
                    "to_state":     curr,
                    "delta_score":  delta,
                    "evidence":     self._evidence_history[i]
                })

        # Sort by delta magnitude, return top n
        moments.sort(key=lambda x: x['delta_score'], reverse=True)
        return moments[:n]
