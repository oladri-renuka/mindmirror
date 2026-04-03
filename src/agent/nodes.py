"""
Phase 4: LangGraph node functions.

Four nodes:
1. window_analyzer   — decides if nudge needed
2. context_builder   — constructs structured prompt
3. llm_nudge         — calls LLM, gets nudge text
4. session_synthesizer — generates final report at end of session

LLM_PROVIDER is auto-detected:
  - HF Spaces (SPACE_ID set) → "hf"  (HF Inference API, free, uses HF_TOKEN)
  - Local dev                → "ollama" (localhost Ollama)
  Override with LLM_PROVIDER env var. Set to "gemini" if you have an API key.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import re
import time
import logging
from typing import Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── LLM Provider Configuration ────────────────────────────────────────────────
# Auto-detect: use HF Inference API on HF Spaces, Ollama locally.
_IS_HF = bool(os.environ.get("SPACE_ID"))
_default_provider = "hf" if _IS_HF else "ollama"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", _default_provider)

# Ollama config (local dev)
OLLAMA_MODEL    = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"

# HF Inference API config (free on HF Spaces)
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Gemini config (optional, if API key available)
GEMINI_MODEL = "gemini-2.0-flash"

# Minimum seconds between nudges (don't over-coach)
MIN_NUDGE_INTERVAL = 30.0

# Consecutive windows in negative state before nudging
NUDGE_THRESHOLD_WINDOWS = 2

# Consecutive CONFIDENT windows for milestone
MILESTONE_THRESHOLD = 4

logger.info(f"LLM provider: {LLM_PROVIDER}")


def _call_llm(prompt: str, max_tokens: int = 100) -> str:
    """
    Call whichever LLM provider is configured.
    Returns generated text or empty string on failure.
    """
    if LLM_PROVIDER == "gemini":
        return _call_gemini(prompt, max_tokens)
    elif LLM_PROVIDER == "hf":
        return _call_hf_inference(prompt, max_tokens)
    else:
        return _call_ollama(prompt, max_tokens)


def _call_ollama(prompt: str, max_tokens: int = 100) -> str:
    """Call local Ollama server."""
    try:
        import ollama
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": max_tokens}
        )
        return response['message']['content'].strip()
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return ""


def _call_hf_inference(prompt: str, max_tokens: int = 100) -> str:
    """
    Call HuggingFace Inference API (free tier).
    Uses HF_TOKEN already required for session storage — no extra secret needed.
    Model: Llama-3.1-8B-Instruct (free, no rate limit for Spaces).
    """
    try:
        from huggingface_hub import InferenceClient
        token  = os.environ.get("HF_TOKEN", "")
        client = InferenceClient(model=HF_MODEL, token=token or None)
        result = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return result.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"HF Inference API call failed: {e}")
        return ""


def _call_gemini(prompt: str, max_tokens: int = 100) -> str:
    """Call Gemini API."""
    try:
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini call failed: {e}")
        return ""


# ── Node 1: Window Analyzer ───────────────────────────────────────────────────

def window_analyzer_node(state: dict) -> dict:
    """
    Analyzes the current window and decides whether to nudge.

    Decision logic:
    - NUDGE if: negative state persisted for N windows
                AND enough time since last nudge
                AND state is NOT THINKING (don't interrupt deliberation)
    - MILESTONE if: CONFIDENT for M consecutive windows
    - Neither: return state unchanged

    This node never calls the LLM — it just sets should_nudge flag.
    """
    state = dict(state)  # Don't mutate input

    current_state = state.get("current_state", "NEUTRAL")
    state_history = state.get("state_history", [])
    last_nudge    = state.get("last_intervention_time", 0.0)
    now           = time.time()

    # Reset decisions from previous cycle
    state["should_nudge"]  = False
    state["is_milestone"]  = False
    state["nudge_text"]    = None
    state["built_prompt"]  = None

    # Never nudge during THINKING — respect deliberate pauses
    if current_state == "THINKING":
        logger.debug("THINKING state — skipping nudge")
        return state

    # Check time since last nudge
    time_since_nudge = now - last_nudge
    if time_since_nudge < MIN_NUDGE_INTERVAL:
        logger.debug(
            f"Too soon since last nudge "
            f"({time_since_nudge:.0f}s < {MIN_NUDGE_INTERVAL}s)"
        )
        return state

    # Check for MILESTONE (consecutive confident windows)
    if len(state_history) >= MILESTONE_THRESHOLD:
        recent = state_history[-MILESTONE_THRESHOLD:]
        if all(s == "CONFIDENT" for s in recent):
            state["should_nudge"] = True
            state["is_milestone"] = True
            logger.info("MILESTONE: sustained confident performance")
            return state

    # Check for negative state persistence
    negative_states = {"NERVOUS", "UNCERTAIN", "DISENGAGED"}
    if current_state in negative_states:
        if len(state_history) >= NUDGE_THRESHOLD_WINDOWS:
            recent = state_history[-NUDGE_THRESHOLD_WINDOWS:]
            if all(s in negative_states for s in recent):
                state["should_nudge"] = True
                logger.info(
                    f"NUDGE triggered: {current_state} "
                    f"persisted for {NUDGE_THRESHOLD_WINDOWS} windows"
                )

    return state


# ── Node 2: Context Builder ───────────────────────────────────────────────────

def context_builder_node(state: dict) -> dict:
    """
    Builds the structured prompt for the LLM.

    This is the most important function in Phase 4.
    The quality of the prompt determines the quality of the nudge.

    Key principle: Gemini/Ollama receives quantitative evidence,
    not just state labels. "eye contact 43% vs baseline 95%" is
    more useful than "poor eye contact".
    """
    state = dict(state)

    if not state.get("should_nudge", False):
        return state

    current_state    = state.get("current_state",    "NEUTRAL")
    current_evidence = state.get("current_evidence", {})
    current_delta    = state.get("current_delta",    "STABLE")
    baseline         = state.get("baseline")         or {}
    session_stats    = state.get("session_stats")    or {}
    is_milestone     = state.get("is_milestone",     False)
    intervention_history = state.get("intervention_history", [])
    question         = state.get("current_question", "general interview question")
    q_num            = state.get("question_number",  1)
    total_q          = state.get("total_questions",  5)

    # Get current fused window data
    fused = state.get("current_fused", {})
    facial = fused.get("facial", {}) if fused else {}
    audio  = fused.get("audio",  {}) if fused else {}

    # Get previous nudge texts to avoid repetition
    prev_nudges = [
        i.get("nudge_text", "")
        for i in intervention_history[-2:]
    ]
    prev_nudge_str = (
        f"\nPrevious nudges given (don't repeat): {prev_nudges}"
        if prev_nudges else ""
    )

    # Weak patterns for session context
    weak_patterns = session_stats.get("weak_patterns", [])
    patterns_str  = (
        f"\nKnown session patterns: {weak_patterns}"
        if weak_patterns else ""
    )

    # ── Milestone prompt ──────────────────────────────────────────────
    if is_milestone:
        prompt = f"""You are a warm, professional interview coach.

The candidate has been performing confidently for the last {MILESTONE_THRESHOLD} consecutive windows.

Session context:
- Question {q_num} of {total_q}: "{question}"
- Eye contact: {session_stats.get('eye_contact_pct', 0):.0%}
- Speaking rate: {session_stats.get('avg_speaking_rate', 0):.0f} wpm
- Trend: {current_delta}
{prev_nudge_str}

Generate ONE positive reinforcement message.
Requirements:
- Under 20 words
- Warm and specific
- Acknowledge what they are doing well
- Do not give any criticism

Respond with ONLY the message, nothing else."""

    # ── Coaching nudge prompt ─────────────────────────────────────────
    else:
        signals   = current_evidence.get("signals",  [])
        metrics   = current_evidence.get("metrics",  {})
        transcript = audio.get("transcript", "")

        # Build evidence string from actual computed metrics
        evidence_lines = []

        eye_open = facial.get("avg_ear", 0)
        if eye_open > 0:
            evidence_lines.append(
                f"- eye_openness: {eye_open:.2f} "
                f"(baseline: {baseline.get('avg_ear', 0.85):.2f})"
            )

        eye_contact = facial.get("eye_contact", True)
        evidence_lines.append(
            f"- eye_contact: {'YES' if eye_contact else 'NO'}"
        )

        rate = audio.get("speaking_rate_wpm", 0)
        if rate > 0:
            evidence_lines.append(
                f"- speaking_rate: {rate:.0f} wpm "
                f"(baseline: {baseline.get('speaking_rate_wpm', 130):.0f} wpm)"
            )

        fillers = audio.get("filler_count", 0)
        if fillers > 0:
            evidence_lines.append(
                f"- filler_words: {fillers} "
                f"({', '.join(audio.get('fillers_detected', [])[:3])})"
            )

        hedges = audio.get("hedge_count", 0)
        if hedges > 0:
            evidence_lines.append(
                f"- hedge_phrases: {hedges} "
                f"({', '.join(audio.get('hedges_detected', [])[:3])})"
            )

        brow = facial.get("brow_stress", 0)
        if brow > 0.3:
            evidence_lines.append(
                f"- brow_stress: {brow:.2f} "
                f"(baseline: {baseline.get('brow_stress', 0.15):.2f})"
            )

        evidence_str = "\n".join(evidence_lines) if evidence_lines else "- general performance"

        prompt = f"""You are a warm, professional interview coach giving real-time feedback.

Current behavioral state: {current_state} (confidence: {state.get('current_confidence', 0):.0%})
Trend: {current_delta}

Evidence (computed from webcam + microphone):
{evidence_str}

Current transcript: "{transcript[:100]}"

Session context:
- Question {q_num} of {total_q}: "{question}"
- Session patterns: {weak_patterns if weak_patterns else 'none yet'}
{prev_nudge_str}
{patterns_str}

Generate ONE specific actionable coaching nudge.
Requirements:
- Under 20 words
- Warm and encouraging (never harsh)
- Specific to the evidence above (reference actual numbers)
- Immediately actionable during this answer
- Do NOT repeat previous nudges

Respond with ONLY the nudge text, nothing else."""

    state["built_prompt"] = prompt
    return state


# ── Node 3: LLM Nudge ─────────────────────────────────────────────────────────

def llm_nudge_node(state: dict) -> dict:
    """
    Calls the LLM with the built prompt.
    Stores nudge text and logs intervention.
    """
    state = dict(state)

    if not state.get("should_nudge") or not state.get("built_prompt"):
        return state

    prompt    = state["built_prompt"]
    is_milestone = state.get("is_milestone", False)

    # Call LLM
    max_tokens = 60 if is_milestone else 40
    nudge_text = _call_llm(prompt, max_tokens=max_tokens)

    if not nudge_text:
        logger.warning("LLM returned empty nudge — using fallback")
        nudge_text = _fallback_nudge(state.get("current_state", "NEUTRAL"))

    # Clean up nudge text
    nudge_text = nudge_text.strip().strip('"').strip("'")
    if len(nudge_text) > 150:
        nudge_text = nudge_text[:147] + "..."

    state["nudge_text"] = nudge_text

    # Log intervention
    intervention = {
        "timestamp":    time.time(),
        "state":        state.get("current_state"),
        "nudge_text":   nudge_text,
        "is_milestone": is_milestone,
        "delta":        state.get("current_delta")
    }

    history = list(state.get("intervention_history", []))
    history.append(intervention)

    state["intervention_history"]   = history
    state["last_intervention_time"] = time.time()
    state["intervention_count"]     = state.get("intervention_count", 0) + 1

    logger.info(f"Nudge generated: '{nudge_text}'")
    return state


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _compute_delivery_score(stats: dict, baseline: dict) -> float:
    """
    Deterministic delivery score (0–10) from behavioral signals.

    Rubric:
      Eye contact      0–3 pts   (target >65%)
      Confident state  0–3 pts   (target >50% of windows)
      Filler words     0–2 pts   (0 fillers = 2, -0.5 per filler)
      Speaking rate    0–2 pts   (100–180 wpm ideal)
    """
    score = 0.0

    # Eye contact (0–3)
    ec = stats.get("eye_contact_pct", 0.0)
    score += min(3.0, (ec / 0.65) * 3.0)

    # Confident state % of windows (0–3)
    dist          = stats.get("state_distribution", {})
    total_windows = max(stats.get("window_count", 1), 1)
    confident_pct = dist.get("CONFIDENT", 0) / total_windows
    score += min(3.0, (confident_pct / 0.50) * 3.0)

    # Filler words (0–2)
    fillers = stats.get("total_fillers", 0)
    score  += max(0.0, 2.0 - fillers * 0.5)

    # Speaking rate (0–2)
    avg_rate = stats.get("avg_speaking_rate", 0.0)
    if 100 <= avg_rate <= 180:
        score += 2.0
    elif avg_rate > 0:
        score += 1.0

    return round(min(10.0, max(0.0, score)), 1)


def _score_content_relevance(question: str, transcript: str) -> Tuple[float, str]:
    """
    Ask the LLM to score content relevance 0–10.

    Returns (score, one_line_feedback).
    Returns (0.0, reason) if transcript is too short to evaluate.
    """
    word_count = len(transcript.split()) if transcript else 0
    if word_count < 15:
        return 0.0, "Too little speech detected to evaluate content relevance."

    prompt = f"""You are a strict interview evaluator. Score only the CONTENT of this answer — not how it was delivered.

QUESTION ASKED: "{question}"

CANDIDATE'S ANSWER:
"{transcript}"

Rules:
- Score 0–2   if the answer is completely off-topic or doesn't address the question at all
- Score 3–5   if the answer is partially relevant but misses the core of the question
- Score 6–8   if the answer addresses the question with reasonable structure
- Score 9–10  if the answer directly, completely, and specifically answers the question

Respond in EXACTLY this format (no other text):
CONTENT_SCORE: <integer 0-10>
CONTENT_FEEDBACK: <one sentence explanation>"""

    response = _call_llm(prompt, max_tokens=80)

    if not response:
        return 5.0, "Content could not be evaluated."

    score_match    = re.search(r'CONTENT_SCORE:\s*(\d+)',  response)
    feedback_match = re.search(r'CONTENT_FEEDBACK:\s*(.+)', response)

    score    = int(score_match.group(1).strip())    if score_match    else 5
    feedback = feedback_match.group(1).strip()      if feedback_match else "Content evaluated."

    score = max(0, min(10, score))
    return float(score), feedback


# ── Node 4: Session Synthesizer ───────────────────────────────────────────────

def session_synthesizer_node(state: dict) -> dict:
    """
    Generates the final session report.
    Called once when user clicks End Session.

    Builds a comprehensive prompt with full session data
    and asks LLM to generate a structured coaching report.
    """
    state = dict(state)

    session_stats = state.get("session_stats") or {}
    baseline      = state.get("baseline")      or {}
    state_history = state.get("state_history") or []
    interventions = state.get("intervention_history", [])
    question      = state.get("current_question", "interview practice")

    # Build state progression string
    progression = " → ".join(state_history[-10:]) if state_history else "No data"

    # State distribution
    from collections import Counter
    dist = Counter(state_history)
    dist_str = ", ".join(
        f"{s}: {c}/{len(state_history)} windows"
        for s, c in dist.most_common()
    )

    # Intervention summary
    nudge_count = len(interventions)
    nudge_texts = [i.get("nudge_text", "") for i in interventions]

    # Session duration
    elapsed = session_stats.get("session_elapsed", 0)
    minutes = int(elapsed // 60)
    seconds = int(elapsed  % 60)

    # ── Two-axis scoring ──────────────────────────────────────────────
    delivery_score = _compute_delivery_score(session_stats, baseline)

    full_transcript = state.get("full_transcript", "")
    content_score, content_feedback = _score_content_relevance(
        question, full_transcript
    )

    # Multiplicative: irrelevant answer tanks the score regardless of delivery
    final_score = round((delivery_score / 10.0) * (content_score / 10.0) * 10.0, 1)

    logger.info(
        f"Scores — delivery: {delivery_score}/10  "
        f"content: {content_score}/10  final: {final_score}/10"
    )

    report_prompt = f"""You are a professional interview coach writing a post-session coaching report.

SESSION DATA:
- Duration: {minutes}m {seconds}s
- Question practiced: "{question}"
- Total windows analyzed: {session_stats.get('window_count', 0)}

BEHAVIORAL ANALYSIS:
- State distribution: {dist_str}
- State progression: {progression}
- Dominant state: {session_stats.get('dominant_state', 'NEUTRAL')}

COMMUNICATION METRICS:
- Eye contact: {session_stats.get('eye_contact_pct', 0):.0%} of session
- Average speaking rate: {session_stats.get('avg_speaking_rate', 0):.0f} wpm
  (baseline: {baseline.get('speaking_rate_wpm', 130):.0f} wpm)
- Total filler words: {session_stats.get('total_fillers', 0)}
- Total hedge phrases: {session_stats.get('total_hedges', 0)}

COACHING INTERVENTIONS:
- Total nudges given: {nudge_count}
- Nudges: {nudge_texts[:5]}

IDENTIFIED PATTERNS:
- {session_stats.get('weak_patterns', ['none identified'])}

SCORING (already computed — use these exact values):
- Delivery score: {delivery_score}/10 (eye contact, confidence, fluency)
- Content score: {content_score}/10 ({content_feedback})
- Final score: {final_score}/10 (delivery × content, combined)

Write a coaching report with these exact sections:
1. OVERALL PERFORMANCE (2 sentences)
2. TOP STRENGTH (1 sentence, specific)
3. TOP IMPROVEMENT AREA (1 sentence — if content score is low, focus on answering the question)
4. RECOMMENDED EXERCISE (2 sentences, actionable)
5. PROGRESS SCORE
{final_score}/10 — Delivery {delivery_score}/10, Content {content_score}/10. [one sentence explanation]

Be warm, specific, and encouraging. Reference actual numbers from the data."""

    logger.info("Generating final session report...")
    report_text = _call_llm(report_prompt, max_tokens=450)

    if not report_text:
        report_text = _fallback_report(session_stats, baseline, delivery_score, content_score, final_score)

    state["final_report"]    = report_text
    state["delivery_score"]  = delivery_score
    state["content_score"]   = content_score
    state["final_score"]     = final_score
    state["session_ended"]   = True

    logger.info("Session report generated")
    return state


# ── Fallback functions ────────────────────────────────────────────────────────

def _fallback_nudge(state: str) -> str:
    """Rule-based fallback if LLM is unavailable."""
    fallbacks = {
        "NERVOUS":    "Take a breath — slow down slightly and look at the camera.",
        "UNCERTAIN":  "You've got this — state your answer directly and confidently.",
        "DISENGAGED": "Re-engage — speak with more energy and maintain eye contact.",
        "CONFIDENT":  "Excellent — keep this energy going!",
        "NEUTRAL":    "Stay focused — you're doing well."
    }
    return fallbacks.get(state, "Keep going — you're doing great!")


def _fallback_report(
    stats: dict,
    baseline: dict,
    delivery_score: float = 5.0,
    content_score:  float = 5.0,
    final_score:    float = 2.5,
) -> str:
    """Rule-based fallback report if LLM is unavailable."""
    ec   = stats.get("eye_contact_pct",   0)
    fil  = stats.get("total_fillers",     0)
    dom  = stats.get("dominant_state",    "NEUTRAL")

    return f"""OVERALL PERFORMANCE
Your dominant state was {dom} with {ec:.0%} eye contact maintained.

TOP STRENGTH
{"Strong eye contact throughout the session." if ec > 0.7 else "Consistent engagement with the practice session."}

TOP IMPROVEMENT AREA
{"Reduce filler words — " + str(fil) + " detected this session. Aim for zero." if fil > 3 else "Ensure your answer directly addresses the question asked."}

RECOMMENDED EXERCISE
Practice the 3-second pause technique before answering each question.
Record yourself and count filler words — awareness is the first step.

PROGRESS SCORE
{final_score}/10 — Delivery {delivery_score}/10, Content {content_score}/10."""
