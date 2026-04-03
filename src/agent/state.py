"""
Phase 4: LangGraph session state definition.

SessionState flows through all agent nodes and accumulates
data across the entire session.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from typing import TypedDict, List, Optional, Any


class SessionState(TypedDict):
    """Complete session state passed between LangGraph nodes."""

    # Current window
    current_state:       str
    current_confidence:  float
    current_evidence:    dict
    current_fused:       Any        # FusedWindow
    current_delta:       str        # IMPROVING/REGRESSING/STABLE

    # Session history
    behavioral_timeline: List[dict]
    state_history:       List[str]
    session_stats:       dict

    # Baseline
    baseline:             Optional[dict]
    baseline_established: bool

    # Intervention tracking
    intervention_history:    List[dict]
    last_intervention_time:  float
    intervention_count:      int

    # Agent decisions this cycle
    should_nudge:  bool
    nudge_text:    Optional[str]
    is_milestone:  bool
    built_prompt:  Optional[str]

    # Session context
    current_question: str
    question_number:  int
    total_questions:  int
    session_start:    float

    # End of session
    session_ended:  bool
    final_report:   Optional[str]
