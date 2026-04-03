"""
Phase 5: Gradio UI for MindMirror.

Browser-mode design — works locally and on HF Spaces:
  • Webcam feed streamed from the user's browser (no server camera needed)
  • Microphone streamed from the user's browser (no server mic needed)
  • On HF Spaces: HF OAuth login, HF Datasets session storage
  • Locally:      name text-field,  local JSON session storage

IS_HF is True when running inside an HF Space (SPACE_ID env var present).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import time
import logging
import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Download MediaPipe model if missing (needed on HF Spaces) ─────────────────
_MODEL_PATH = "models/face_landmarker.task"
_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(_MODEL_PATH):
    import urllib.request
    os.makedirs("models", exist_ok=True)
    print(f"Downloading MediaPipe face landmarker model...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("Model downloaded.")

from pipeline import MindMirrorPipeline
from src.output.session_logger import SessionLogger

# ── Environment detection ─────────────────────────────────────────────────────
IS_HF    = bool(os.environ.get("SPACE_ID"))
BACKEND  = "hf" if IS_HF else "local"

# ── Global instances ──────────────────────────────────────────────────────────
pipeline       = MindMirrorPipeline()
state_timeline = deque(maxlen=50)   # (elapsed, state) pairs
nudge_history  = deque(maxlen=20)   # (timestamp, text, is_milestone)

# ── State colours ─────────────────────────────────────────────────────────────
STATE_COLORS = {
    "CONFIDENT":  "#22c55e",
    "NERVOUS":    "#ef4444",
    "UNCERTAIN":  "#f59e0b",
    "THINKING":   "#3b82f6",
    "DISENGAGED": "#6b7280",
    "NEUTRAL":    "#d1d5db",
}
STATE_EMOJI = {
    "CONFIDENT":  "🟢",
    "NERVOUS":    "🔴",
    "UNCERTAIN":  "🟡",
    "THINKING":   "🔵",
    "DISENGAGED": "⚫",
    "NEUTRAL":    "⚪",
}


# ── UI helpers ────────────────────────────────────────────────────────────────

def format_timer(seconds: float) -> str:
    m, s = int(seconds // 60), int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def build_state_badge(state: str, confidence: float, delta: str) -> str:
    color  = STATE_COLORS.get(state, "#d1d5db")
    emoji  = STATE_EMOJI.get(state, "⚪")
    arrow  = {"IMPROVING": "↑", "REGRESSING": "↓", "STABLE": "→"}.get(delta, "→")
    dcol   = {"IMPROVING": "#22c55e", "REGRESSING": "#ef4444", "STABLE": "#9ca3af"}.get(delta, "#9ca3af")
    return f"""
<div style="background:{color}22;border:2px solid {color};border-radius:12px;
            padding:16px;text-align:center;font-family:system-ui">
  <div style="font-size:2.5em">{emoji}</div>
  <div style="font-size:1.4em;font-weight:bold;color:{color};margin:4px 0">{state}</div>
  <div style="font-size:0.9em;color:#6b7280">Confidence: {confidence:.0%}</div>
  <div style="font-size:1.1em;color:{dcol};font-weight:bold">{arrow} {delta}</div>
</div>"""


def build_metrics_html(result: dict) -> str:
    ec_pct  = result.get("eye_contact_pct", 0)
    ec_col  = "#22c55e" if ec_pct > 0.6 else "#f59e0b" if ec_pct > 0.3 else "#ef4444"
    rate    = result.get("speaking_rate", 0)
    r_col   = "#22c55e" if 100 <= rate <= 180 else "#f59e0b" if rate > 0 else "#6b7280"
    fil     = result.get("total_fillers", 0)
    f_col   = "#22c55e" if fil == 0 else "#f59e0b" if fil <= 3 else "#ef4444"
    hed     = result.get("total_hedges", 0)
    h_col   = "#22c55e" if hed <= 1 else "#f59e0b" if hed <= 3 else "#ef4444"
    ts      = result.get("transcript", "")
    timer_s = format_timer(result.get("elapsed", 0))

    def row(label, value, color, unit=""):
        return (f'<div style="display:flex;justify-content:space-between;'
                f'padding:6px 0;border-bottom:1px solid #f3f4f6">'
                f'<span style="color:#6b7280;font-size:.85em">{label}</span>'
                f'<span style="color:{color};font-weight:bold">{value}{unit}</span></div>')

    return f"""
<div style="font-family:system-ui;padding:8px">
  <div style="text-align:right;color:#6b7280;font-size:.8em;margin-bottom:8px">⏱ {timer_s}</div>
  {row("👁 Eye Contact",    f"{ec_pct:.0%}", ec_col)}
  {row("🎙 Speaking Rate",  f"{rate:.0f}",  r_col, " wpm")}
  {row("🔤 Filler Words",   fil,            f_col, " total")}
  {row("💭 Hedge Phrases",  hed,            h_col, " total")}
  {row("📝 Windows",        result.get("window_count", 0), "#6b7280")}
  <div style="margin-top:10px;padding:8px;background:#f9fafb;border-radius:6px;
              font-size:.8em;color:#374151;font-style:italic;min-height:40px">
    "{ts[:80]}{'...' if len(ts) > 80 else ''}"
  </div>
</div>"""


def build_nudge_html(nudge_text: str, is_milestone: bool) -> str:
    if not nudge_text:
        return """<div style="background:#f9fafb;border:1px dashed #d1d5db;border-radius:10px;
                              padding:16px;text-align:center;color:#9ca3af;font-family:system-ui;
                              font-style:italic">Coaching nudges will appear here during your session</div>"""
    bg, border, icon, label = (
        ("#f0fdf4", "#22c55e", "🌟", "Milestone") if is_milestone
        else ("#eff6ff", "#3b82f6", "💬", "Coach")
    )
    return f"""<div style="background:{bg};border-left:4px solid {border};border-radius:0 10px 10px 0;
                            padding:16px 20px;font-family:system-ui">
  <div style="font-size:.75em;color:#6b7280;margin-bottom:4px">{icon} {label}</div>
  <div style="font-size:1.05em;color:#1f2937;font-weight:500">{nudge_text}</div>
</div>"""


def build_emotion_timeline_chart(timeline: list):
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f9fafb")

    if not timeline:
        ax.text(0.5, 0.5, "Timeline will appear after session starts",
                ha="center", va="center", color="#9ca3af", fontsize=11, transform=ax.transAxes)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        plt.tight_layout(); return fig

    states     = [t[1] for t in timeline]
    timestamps = [t[0] for t in timeline]
    t0         = timestamps[0]
    times      = [t - t0 for t in timestamps]

    for t, state in zip(times, states):
        ax.barh(0, 2.0, left=t, height=0.6, color=STATE_COLORS.get(state, "#d1d5db"),
                alpha=0.85, edgecolor="white", linewidth=0.5)

    patches = [mpatches.Patch(color=STATE_COLORS[s], label=s)
               for s in STATE_COLORS if s in states]
    if patches:
        ax.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.9, ncol=len(patches))

    total = times[-1] + 2 if times else 10
    ax.set_xlim(0, max(total + 2, 10)); ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Time (seconds)", fontsize=9, color="#6b7280"); ax.set_yticks([])
    ax.set_title("Emotion Timeline", fontsize=10, color="#374151", pad=8)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout(); return fig


# ── History helpers ───────────────────────────────────────────────────────────

def build_progress_chart(sessions: list):
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#ffffff"); ax.set_facecolor("#f9fafb")

    if not sessions:
        ax.text(0.5, 0.5, "No sessions yet — complete a session to see your progress",
                ha="center", va="center", color="#9ca3af", fontsize=11, transform=ax.transAxes)
        ax.axis("off"); plt.tight_layout(); return fig

    chrono = list(reversed(sessions))
    labels = [s.get("timestamp", "")[:10] for s in chrono]
    scores = [s.get("progress_score") for s in chrono]
    valid  = [(i, s) for i, s in enumerate(scores) if s is not None]

    if not valid:
        ax.text(0.5, 0.5, "No scores available yet",
                ha="center", va="center", color="#9ca3af", fontsize=11, transform=ax.transAxes)
        ax.axis("off"); plt.tight_layout(); return fig

    idxs, vals = zip(*valid)
    ax.plot(list(idxs), list(vals), "o-", color="#3b82f6", linewidth=2, markersize=8)
    ax.fill_between(list(idxs), list(vals), alpha=0.1, color="#3b82f6")
    for x, y in zip(idxs, vals):
        ax.annotate(f"{y}/10", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color="#1f2937")

    ax.set_ylim(0, 11)
    ax.set_xticks(list(range(len(chrono))))
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Score", fontsize=9, color="#6b7280")
    ax.set_title("Progress Score Over Time", fontsize=10, color="#374151", pad=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); return fig


def load_history(username: str = ""):
    sessions = SessionLogger(username=username or "anonymous", backend=BACKEND).load_all()
    if not sessions:
        return [], build_progress_chart([])

    rows = []
    for i, s in enumerate(sessions, 1):
        date     = s.get("timestamp", "")[:16].replace("T", " ")
        q        = s.get("question",  "")[:45]
        final    = s.get("progress_score")
        delivery = s.get("delivery_score")
        content  = s.get("content_score")
        score_s  = (
            f"{final}/10  (D:{delivery} C:{content})"
            if final is not None and delivery is not None
            else f"{final}/10" if final is not None else "N/A"
        )
        dist = s.get("state_distribution", {})
        dom  = max(dist, key=dist.get) if dist else "N/A"
        dur  = format_timer(s.get("duration_seconds", 0))
        rows.append([i, date, q, score_s, dom, dur])

    return rows, build_progress_chart(sessions)


def view_session_report(session_num: int, username: str = "") -> str:
    sessions = SessionLogger(username=username or "anonymous", backend=BACKEND).load_all()
    if not sessions:
        return "No sessions found."
    idx = int(session_num) - 1
    if idx < 0 or idx >= len(sessions):
        return f"Session {int(session_num)} not found — you have {len(sessions)} session(s)."
    s        = sessions[idx]
    final    = s.get("progress_score")
    delivery = s.get("delivery_score")
    content  = s.get("content_score")
    score_line = (
        f"Score:    {final}/10  (Delivery {delivery}/10 × Content {content}/10)"
        if final is not None and delivery is not None
        else f"Score:    {final}/10" if final is not None else "Score:    N/A"
    )
    meta = (
        f"Date:     {s.get('timestamp', '')[:16].replace('T', ' ')}\n"
        f"Question: {s.get('question', '')}\n"
        f"Duration: {format_timer(s.get('duration_seconds', 0))}\n"
        f"{score_line}\n" + "─" * 44 + "\n"
    )
    return meta + s.get("report", "No report available.")


# ── Stream callbacks ──────────────────────────────────────────────────────────

def on_video_stream(frame: np.ndarray):
    """Receives browser webcam frame, pushes to pipeline for MediaPipe processing."""
    if frame is not None:
        pipeline.push_video_frame(frame)


def on_audio_stream(audio):
    """Receives browser mic chunk, pushes to pipeline audio buffer."""
    if audio is not None and pipeline.is_active():
        sample_rate, data = audio
        pipeline.push_audio_chunk(sample_rate, data)


# ── Session event handlers ────────────────────────────────────────────────────

def start_session(
    question: str,
    name: str,
    oauth_profile: gr.OAuthProfile | None = None,
):
    global state_timeline, nudge_history

    # Resolve username — OAuth on HF, name input locally
    if IS_HF:
        if oauth_profile is None:
            return (
                gr.update(value="⚠️ Please sign in with HuggingFace first"),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=True),
                build_nudge_html(None, False),
                "",
                build_emotion_timeline_chart([]),
                "",
            )
        username = oauth_profile.username
    else:
        username = name.strip()
        if not username:
            return (
                gr.update(value="⚠️ Please enter your name before starting"),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=True),
                build_nudge_html(None, False),
                "",
                build_emotion_timeline_chart([]),
                "",
            )

    if not question.strip():
        question = "Tell me about yourself"

    state_timeline.clear()
    nudge_history.clear()

    pipeline.start_session(question=question, question_number=1,
                           total_questions=1, username=username)

    return (
        gr.update(value=f"✅ Session started — '{question}' ({username})"),
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(interactive=False),
        build_nudge_html(None, False),
        "",
        build_emotion_timeline_chart([]),
        username,
    )


def end_session():
    if not pipeline.is_active():
        return (
            gr.update(value="No active session"),
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=True),
            "No active session to end.",
            build_emotion_timeline_chart(list(state_timeline)),
        )

    report = pipeline.end_session()
    return (
        gr.update(value="✅ Session complete — see report below"),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=True),
        report,
        build_emotion_timeline_chart(list(state_timeline)),
    )


def update_dashboard():
    """Called every 2 seconds by Gradio Timer. Does NOT handle webcam display."""
    result = pipeline.process_tick()

    if result.get("phase") == "coaching" and result.get("state"):
        state_timeline.append((result.get("elapsed", 0), result.get("state", "NEUTRAL")))

    nudge_text   = result.get("nudge_text")
    is_milestone = result.get("is_milestone", False)
    if nudge_text:
        nudge_history.append((time.time(), nudge_text, is_milestone))

    recent_nudge, recent_ms = None, False
    if nudge_history:
        last_ts, last_nudge, last_ms = nudge_history[-1]
        if time.time() - last_ts < 30:
            recent_nudge, recent_ms = last_nudge, last_ms

    chart = build_emotion_timeline_chart(list(state_timeline))

    if result.get("phase") == "calibrating":
        pct = result.get("calibration_pct", 0)
        calibrating_html = f"""
<div style="font-family:system-ui;text-align:center;padding:20px">
  <div style="font-size:1.1em;color:#6b7280;margin-bottom:8px">🔄 Calibrating your baseline...</div>
  <div style="background:#e5e7eb;border-radius:99px;height:8px;margin:8px 0">
    <div style="background:#3b82f6;width:{pct*100:.0f}%;height:8px;border-radius:99px;transition:width .3s"></div>
  </div>
  <div style="color:#9ca3af;font-size:.8em">{result.get("status_text", "")}</div>
</div>"""
        return (calibrating_html, build_metrics_html(result),
                build_nudge_html(None, False), chart)

    return (
        build_state_badge(result.get("state", "NEUTRAL"),
                          result.get("confidence", 0.0),
                          result.get("delta", "STABLE")),
        build_metrics_html(result),
        build_nudge_html(recent_nudge, recent_ms),
        chart,
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="MindMirror — AI Interview Coach") as demo:

    # ── Header ────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center;padding:20px 0 10px;font-family:system-ui">
      <h1 style="font-size:2em;font-weight:bold;color:#1f2937;margin:0">🪞 MindMirror</h1>
      <p style="color:#6b7280;margin:4px 0 0">Real-time AI Interview Coach</p>
    </div>""")

    # ── Auth row ──────────────────────────────────────────────────────
    with gr.Row():
        if IS_HF:
            gr.LoginButton(scale=1)
            gr.LogoutButton(scale=1)
            user_info = gr.Markdown("Sign in with HuggingFace to track your progress →", scale=4)
        else:
            name_input = gr.Textbox(
                label="Your Name",
                placeholder="Enter your name to track your progress",
                scale=3,
            )
            gr.HTML('<div style="scale:0"></div>', scale=3)   # spacer

    current_user = gr.State(value="")

    # ── Session controls ───────────────────────────────────────────────
    with gr.Row():
        question_input = gr.Textbox(
            label="Interview Question",
            placeholder="e.g. Tell me about a challenge you overcame",
            value="Tell me about yourself",
            scale=4,
        )
        start_btn = gr.Button("▶ Start Session", variant="primary", scale=1)
        end_btn   = gr.Button("⏹ End Session",   variant="stop",    scale=1, interactive=False)

    status_bar = gr.Textbox(
        value="Ready — allow camera & mic access, then click Start Session",
        label="Status", interactive=False, max_lines=1,
    )

    # ── Main dashboard ────────────────────────────────────────────────
    with gr.Row():
        # Left: webcam (browser streaming — handles display automatically)
        with gr.Column(scale=5):
            webcam_display = gr.Image(
                sources=["webcam"],
                streaming=True,
                label="Live Feed",
                height=360,
            )

        # Right: state + metrics
        with gr.Column(scale=5):
            state_display   = gr.HTML(value=build_state_badge("NEUTRAL", 0.0, "STABLE"))
            metrics_display = gr.HTML(value=build_metrics_html(pipeline._empty_result()))

    # ── Microphone instruction banner ─────────────────────────────────
    gr.HTML("""
    <div style="background:#fffbeb;border:1px solid #fbbf24;border-radius:8px;
                padding:10px 16px;font-family:system-ui;font-size:.9em;color:#92400e;margin:4px 0">
      <strong>⚠️ Step 1:</strong> Click the microphone button below and allow access &nbsp;|&nbsp;
      <strong>Step 2:</strong> Allow webcam access above &nbsp;|&nbsp;
      <strong>Step 3:</strong> Click <em>Start Session</em>
    </div>""")

    # ── Microphone input (streams audio to pipeline) ───────────────────
    mic_input = gr.Audio(
        sources=["microphone"],
        streaming=True,
        label="🎙 Microphone — activate this BEFORE clicking Start Session",
        type="numpy",
    )

    # ── Coaching nudge ────────────────────────────────────────────────
    gr.HTML("<hr style='border:none;border-top:1px solid #f3f4f6;margin:8px 0'>")
    nudge_display = gr.HTML(value=build_nudge_html(None, False))

    # ── Emotion timeline ──────────────────────────────────────────────
    gr.HTML("<hr style='border:none;border-top:1px solid #f3f4f6;margin:8px 0'>")
    timeline_chart = gr.Plot(label="Emotion Timeline", show_label=True)

    # ── Session report ────────────────────────────────────────────────
    gr.HTML("<hr style='border:none;border-top:1px solid #f3f4f6;margin:8px 0'>")
    report_display = gr.Textbox(
        label="📋 Session Report", lines=15, interactive=False,
        placeholder="Your coaching report will appear here after ending the session...",
    )

    # ── History section ───────────────────────────────────────────────
    gr.HTML("<hr style='border:none;border-top:2px solid #e5e7eb;margin:16px 0'>")
    gr.HTML('<div style="font-family:system-ui;font-size:1.1em;font-weight:600;color:#1f2937;padding:4px 0 8px">📈 Session History</div>')

    with gr.Row():
        refresh_history_btn = gr.Button("🔄 Refresh", variant="secondary", scale=1)
        session_num_input   = gr.Number(
            value=1, label="View report # (1 = most recent)",
            minimum=1, step=1, precision=0, scale=2,
        )
        view_report_btn = gr.Button("📋 View Report", variant="secondary", scale=1)

    progress_chart  = gr.Plot(label="Progress Score Over Time")
    history_table   = gr.Dataframe(
        headers=["#", "Date", "Question", "Score", "Dominant State", "Duration"],
        datatype=["number", "str", "str", "str", "str", "str"],
        label="Past Sessions", interactive=False, wrap=True,
    )
    history_report  = gr.Textbox(
        label="Selected Session Report", lines=12, interactive=False,
        placeholder="Click 'View Report' to load a session...",
    )

    # ── 2-second timer ────────────────────────────────────────────────
    timer = gr.Timer(value=2)

    # ── Event wiring ──────────────────────────────────────────────────

    # Webcam stream → pipeline facial buffer only (display is handled by the component)
    webcam_display.stream(
        fn=on_video_stream,
        inputs=[webcam_display],
        outputs=[],
        stream_every=0.15,   # ~7 fps — sufficient for face analysis
    )

    # Mic stream → pipeline audio buffer
    mic_input.stream(
        fn=on_audio_stream,
        inputs=[mic_input],
        outputs=[],
    )

    # Start / End buttons
    start_btn.click(
        fn=start_session,
        inputs=[question_input, name_input if not IS_HF else gr.State(value="")],
        outputs=[status_bar, start_btn, end_btn, question_input,
                 nudge_display, report_display, timeline_chart, current_user],
    )

    end_btn.click(
        fn=end_session,
        inputs=[],
        outputs=[status_bar, start_btn, end_btn, question_input,
                 report_display, timeline_chart],
    ).then(
        fn=load_history,
        inputs=[current_user],
        outputs=[history_table, progress_chart],
    )

    # Timer tick — state/metrics/nudge/timeline only (webcam handled by stream)
    timer.tick(
        fn=update_dashboard,
        inputs=[],
        outputs=[state_display, metrics_display, nudge_display, timeline_chart],
    )

    # History controls
    refresh_history_btn.click(fn=load_history, inputs=[current_user],
                               outputs=[history_table, progress_chart])
    view_report_btn.click(fn=view_session_report,
                          inputs=[session_num_input, current_user],
                          outputs=[history_report])

    # On HF: show logged-in username on page load
    if IS_HF:
        def _show_user(oauth_profile: gr.OAuthProfile | None):
            if oauth_profile:
                return f"✅ Signed in as **{oauth_profile.username}**"
            return "Sign in with HuggingFace to track your progress →"
        demo.load(fn=_show_user, inputs=None, outputs=[user_info])

    # ── Footer ────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center;padding:16px;color:#9ca3af;font-size:.75em;font-family:system-ui">
      MindMirror — Multimodal AI Interview Coach |
      Vision: MediaPipe · Audio: Whisper · Agent: LangGraph
    </div>""")


# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  MindMirror — AI Interview Coach")
    print("=" * 55)
    _provider = os.getenv("LLM_PROVIDER", "hf" if IS_HF else "ollama")
    print(f"  Mode:         {'HF Spaces' if IS_HF else 'Local dev'}")
    print(f"  LLM Provider: {_provider}")
    if not IS_HF:
        print("  Make sure Ollama is running: ollama serve")
    print("=" * 55 + "\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css="""
        .gradio-container { max-width: 1100px !important; margin: 0 auto }
        footer { display: none !important }
        """,
    )
