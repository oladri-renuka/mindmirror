"""
Phase 4: LangGraph graph wiring.

Graph structure:
  START
    ↓
  window_analyzer    ← runs every 2 seconds
    ↓ (if should_nudge)
  context_builder
    ↓
  llm_nudge
    ↓
  END

  session_synthesizer ← triggered separately at end of session
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import logging
from langgraph.graph import StateGraph, END
from src.agent.state import SessionState
from src.agent.nodes import (
    window_analyzer_node,
    context_builder_node,
    llm_nudge_node,
    session_synthesizer_node
)

logger = logging.getLogger(__name__)


def _should_nudge(state: dict) -> str:
    """Routing function — go to context_builder or end."""
    if state.get("should_nudge", False):
        return "context_builder"
    return END


def build_window_graph():
    """
    Build the 2-second window processing graph.
    Called on every window during the session.
    """
    graph = StateGraph(dict)

    graph.add_node("window_analyzer", window_analyzer_node)
    graph.add_node("context_builder", context_builder_node)
    graph.add_node("llm_nudge",       llm_nudge_node)

    graph.set_entry_point("window_analyzer")

    graph.add_conditional_edges(
        "window_analyzer",
        _should_nudge,
        {
            "context_builder": "context_builder",
            END:                END
        }
    )

    graph.add_edge("context_builder", "llm_nudge")
    graph.add_edge("llm_nudge",       END)

    return graph.compile()


def build_report_graph():
    """
    Build the end-of-session report graph.
    Called once when user ends the session.
    """
    graph = StateGraph(dict)

    graph.add_node("session_synthesizer", session_synthesizer_node)
    graph.set_entry_point("session_synthesizer")
    graph.add_edge("session_synthesizer", END)

    return graph.compile()


# Compile graphs once at import time
WINDOW_GRAPH = build_window_graph()
REPORT_GRAPH = build_report_graph()

logger.info("LangGraph graphs compiled successfully")
