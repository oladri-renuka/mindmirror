"""
Phase 4: Agent runner — clean external interface.

This is the only file pipeline.py imports from the agent module.
Hides all LangGraph complexity behind two simple functions:

    process_window(fused_window, classifier_output, session_stats, state)
    generate_report(state)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import time
import logging
from typing import Optional, Tuple
from src.agent.graph import WINDOW_GRAPH, REPORT_GRAPH
from src.agent.state import SessionState

logger = logging.getLogger(__name__)


def get_initial_state(
    question:        str = "Tell me about yourself",
    question_number: int = 1,
    total_questions: int = 5
) -> dict:
    """
    Create a fresh SessionState for a new session.
    Call this once at the start of each session.
    """
    return {
        # Current window (will be updated each cycle)
        "current_state":       "NEUTRAL",
        "current_confidence":  0.5,
        "current_evidence":    {},
        "current_fused":       None,
        "current_delta":       "STABLE",

        # History (grows over session)
        "behavioral_timeline": [],
        "state_history":       [],
        "session_stats":       {},

        # Baseline (established after first 7 windows)
        "baseline":             None,
        "baseline_established": False,

        # Interventions
        "intervention_history":   [],
        "last_intervention_time": 0.0,
        "intervention_count":     0,

        # Agent decisions
        "should_nudge":  False,
        "nudge_text":    None,
        "is_milestone":  False,
        "built_prompt":  None,

        # Session context
        "current_question": question,
        "question_number":  question_number,
        "total_questions":  total_questions,
        "session_start":    time.time(),

        # End of session
        "session_ended":   False,
        "final_report":    None,
        "full_transcript": "",
        "delivery_score":  None,
        "content_score":   None,
        "final_score":     None,
    }


def process_window(
    fused_window,
    state_label:    str,
    confidence:     float,
    evidence:       dict,
    session_stats:  dict,
    baseline:       Optional[dict],
    agent_state:    dict
) -> Tuple[Optional[str], bool, dict]:
    """
    Process one 2-second window through the agent.

    Args:
        fused_window:   FusedWindow from fusion.py
        state_label:    State from classifier ("NERVOUS" etc.)
        confidence:     Classifier confidence 0-1
        evidence:       Evidence dict from classifier
        session_stats:  Current session stats dict
        baseline:       Personal baseline or None
        agent_state:    Current SessionState dict

    Returns:
        (nudge_text, is_milestone, updated_state)
        nudge_text:   str if nudge generated, None otherwise
        is_milestone: True if this is positive reinforcement
        updated_state: Updated SessionState for next cycle
    """
    # Update state with current window data
    updated = dict(agent_state)
    updated["current_state"]      = state_label
    updated["current_confidence"] = confidence
    updated["current_evidence"]   = evidence
    updated["current_fused"]      = fused_window
    updated["session_stats"]      = session_stats
    updated["baseline"]           = baseline
    updated["baseline_established"] = baseline is not None

    # Compute delta from history
    history = updated.get("state_history", [])
    updated["current_delta"] = _compute_delta(state_label, history)

    # Update state history
    history = list(history)
    history.append(state_label)
    updated["state_history"] = history

    # Update behavioral timeline
    timeline = list(updated.get("behavioral_timeline", []))
    timeline.append({
        "state":      state_label,
        "confidence": confidence,
        "evidence":   evidence,
        "timestamp":  time.time()
    })
    updated["behavioral_timeline"] = timeline

    # Run through LangGraph window graph
    try:
        result = WINDOW_GRAPH.invoke(updated)
    except Exception as e:
        logger.error(f"Window graph failed: {e}")
        return None, False, updated

    nudge_text   = result.get("nudge_text")
    is_milestone = result.get("is_milestone", False)

    return nudge_text, is_milestone, result


def generate_report(agent_state: dict) -> dict:
    """
    Generate the final session report.

    Call this when user ends the session.
    Runs session_synthesizer_node through the report graph.

    Returns dict with keys:
        report          — full report text
        delivery_score  — float 0-10 (behavioral signals)
        content_score   — float 0-10 (answer relevance)
        final_score     — float 0-10 (delivery × content, combined)
    """
    try:
        result = REPORT_GRAPH.invoke(agent_state)
        if result is None:
            raise ValueError("Report graph returned None — session may be too short")
        return {
            "report":         result.get("final_report", "Report generation failed."),
            "delivery_score": result.get("delivery_score"),
            "content_score":  result.get("content_score"),
            "final_score":    result.get("final_score"),
        }
    except Exception as e:
        logger.error(f"Report graph failed: {e}")
        return {
            "report":         f"Session complete. Error generating report: {e}",
            "delivery_score": None,
            "content_score":  None,
            "final_score":    None,
        }


def _compute_delta(current: str, history: list) -> str:
    """Compute trajectory from state history."""
    STATE_SCORE = {
        "DISENGAGED": 0,
        "NERVOUS":    1,
        "UNCERTAIN":  2,
        "NEUTRAL":    3,
        "THINKING":   4,
        "CONFIDENT":  5
    }

    if len(history) < 2:
        return "STABLE"

    current_score = STATE_SCORE.get(current, 3)
    recent        = history[-3:]
    avg_recent    = sum(
        STATE_SCORE.get(s, 3) for s in recent
    ) / len(recent)

    diff = current_score - avg_recent

    if diff >= 1.0:
        return "IMPROVING"
    elif diff <= -1.0:
        return "REGRESSING"
    return "STABLE"
