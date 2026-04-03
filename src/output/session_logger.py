"""
Session persistence for MindMirror.

Two backends:
  "local"  — JSON files in sessions/{username}/  (development)
  "hf"     — HF Datasets repo (production on HF Spaces)

HF backend requires two Space secrets:
  HF_TOKEN          — write-access token for the dataset repo
  HF_DATASET_REPO   — repo ID, e.g. "your-username/mindmirror-sessions"
"""

import io
import json
import logging
import os
import re
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)

SESSIONS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "sessions")
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitize_username(username: str) -> str:
    """Make username safe for use as a directory / repo path component."""
    name = username.strip().lower()
    name = re.sub(r"[^a-z0-9_\-]", "_", name)
    return name or "anonymous"


def _extract_progress_score(report: str) -> Optional[int]:
    """
    Parse the X/10 progress score from report text.

    Handles both inline  ('PROGRESS SCORE: 7/10')
    and multiline        ('PROGRESS SCORE\\n7/10') formats.
    Used only as a fallback when the score is not passed explicitly.
    """
    if not report:
        return None
    match = re.search(r"PROGRESS SCORE[^0-9]*(\d+)\s*/\s*10", report, re.IGNORECASE)
    return int(match.group(1)) if match else None


# ── SessionLogger ─────────────────────────────────────────────────────────────

class SessionLogger:
    """
    Saves and loads session JSON files for one user.

    Usage:
        logger = SessionLogger(username="renuka", backend="local")
        path   = logger.save(question=..., report=..., ...)
        past   = logger.load_all()   # newest first
    """

    def __init__(self, username: str = "anonymous", backend: str = "local"):
        self.backend  = backend
        self.username = _sanitize_username(username)

        if backend == "local":
            self.sessions_dir = os.path.join(SESSIONS_ROOT, self.username)
            os.makedirs(self.sessions_dir, exist_ok=True)

        elif backend == "hf":
            self._hf_token   = os.environ.get("HF_TOKEN", "")
            self._hf_repo    = os.environ.get("HF_DATASET_REPO", "")
            if not self._hf_token or not self._hf_repo:
                logger.warning(
                    "HF backend selected but HF_TOKEN / HF_DATASET_REPO "
                    "env vars are not set — falling back to local storage"
                )
                self.backend      = "local"
                self.sessions_dir = os.path.join(SESSIONS_ROOT, self.username)
                os.makedirs(self.sessions_dir, exist_ok=True)

    # ── save ──────────────────────────────────────────────────────────────────

    def save(
        self,
        question:           str,
        duration_seconds:   float,
        state_timeline:     list,
        state_distribution: dict,
        metrics:            dict,
        nudges:             list,
        report:             str,
        delivery_score:     Optional[float] = None,
        content_score:      Optional[float] = None,
        final_score:        Optional[float] = None,
    ) -> str:
        """Persist session data. Returns file path (local) or repo path (HF)."""
        now        = datetime.now()
        session_id = now.strftime("%Y%m%d_%H%M%S")

        progress_score = (
            final_score if final_score is not None
            else _extract_progress_score(report)
        )

        data = {
            "session_id":         session_id,
            "timestamp":          now.isoformat(),
            "question":           question,
            "duration_seconds":   round(duration_seconds, 1),
            "state_timeline":     state_timeline,
            "state_distribution": state_distribution,
            "metrics":            metrics,
            "nudges":             nudges,
            "report":             report,
            "progress_score":     progress_score,
            "delivery_score":     delivery_score,
            "content_score":      content_score,
        }

        if self.backend == "hf":
            return self._save_hf(data, session_id)
        return self._save_local(data, session_id)

    def _save_local(self, data: dict, session_id: str) -> str:
        filepath = os.path.join(self.sessions_dir, f"session_{session_id}.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return filepath

    def _save_hf(self, data: dict, session_id: str) -> str:
        from huggingface_hub import HfApi
        api          = HfApi(token=self._hf_token)
        path_in_repo = f"{self.username}/session_{session_id}.json"
        json_bytes   = json.dumps(data, indent=2, default=str).encode()

        api.upload_file(
            path_or_fileobj = io.BytesIO(json_bytes),
            path_in_repo    = path_in_repo,
            repo_id         = self._hf_repo,
            repo_type       = "dataset",
            commit_message  = f"session {session_id} for {self.username}",
        )
        logger.info(f"Session saved to HF: {self._hf_repo}/{path_in_repo}")
        return path_in_repo

    # ── load_all ──────────────────────────────────────────────────────────────

    def load_all(self) -> List[dict]:
        """Return all sessions for this user, newest first."""
        if self.backend == "hf":
            return self._load_all_hf()
        return self._load_all_local()

    def _load_all_local(self) -> List[dict]:
        sessions = []
        if not os.path.isdir(self.sessions_dir):
            return sessions
        for fname in sorted(os.listdir(self.sessions_dir), reverse=True):
            if fname.startswith("session_") and fname.endswith(".json"):
                try:
                    with open(os.path.join(self.sessions_dir, fname)) as f:
                        sessions.append(json.load(f))
                except Exception:
                    continue
        return sessions

    def _load_all_hf(self) -> List[dict]:
        from huggingface_hub import HfApi, hf_hub_download
        api      = HfApi(token=self._hf_token)
        sessions = []

        try:
            all_files   = list(api.list_repo_files(repo_id=self._hf_repo, repo_type="dataset"))
            prefix      = f"{self.username}/session_"
            user_files  = sorted(
                [f for f in all_files if f.startswith(prefix) and f.endswith(".json")],
                reverse=True,
            )
            for file_path in user_files:
                try:
                    local = hf_hub_download(
                        repo_id   = self._hf_repo,
                        filename  = file_path,
                        repo_type = "dataset",
                        token     = self._hf_token,
                    )
                    with open(local) as f:
                        sessions.append(json.load(f))
                except Exception as e:
                    logger.warning(f"Could not load {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to list HF sessions: {e}")

        return sessions
