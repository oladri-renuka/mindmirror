"""
Phase 3: Personal baseline establishment.

The first 15 seconds of every session are a calibration period.
We collect FusedWindows and compute the user's natural resting values.

Why this matters:
- Someone who naturally speaks at 180wpm should not be flagged as nervous
  when speaking at 185wpm
- Someone with naturally low pitch variance should not be flagged as disengaged
- Someone with naturally high brow stress (resting face) needs a higher threshold

All subsequent behavioral state classification compares against
this personal baseline rather than population norms.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import logging
import time
from typing import List, Optional
from src.contracts import FusedWindow, SessionBaseline

logger = logging.getLogger(__name__)

# How many windows to collect for baseline
# At 2s per window: 7 windows = 14 seconds of calibration
BASELINE_WINDOW_COUNT = 7

# Minimum windows needed before baseline is considered valid
MIN_BASELINE_WINDOWS = 4


class BaselineEstablisher:
    """
    Collects FusedWindows during the calibration period
    and computes personal baseline values.

    Usage:
        bl = BaselineEstablisher()

        # Feed windows during first 15 seconds
        for window in first_windows:
            bl.add_window(window)
            if bl.is_ready():
                break

        baseline = bl.get_baseline()
        # Use baseline in classifier
    """

    def __init__(self, target_windows: int = BASELINE_WINDOW_COUNT):
        self._windows: List[FusedWindow] = []
        self._target_windows = target_windows
        self._baseline: Optional[SessionBaseline] = None
        self._start_time = time.time()

    def add_window(self, window: FusedWindow):
        """
        Add a FusedWindow to the calibration pool.
        Once target_windows reached, baseline is computed automatically.
        """
        if self._baseline is not None:
            return  # Already established, ignore new windows

        # Only add windows where face was detected
        if not window['facial'].get('face_detected', False):
            logger.debug("Skipping window — no face detected")
            return

        self._windows.append(window)
        logger.debug(
            f"Baseline window {len(self._windows)}/{self._target_windows} collected"
        )

        if len(self._windows) >= self._target_windows:
            self._compute_baseline()

    def is_ready(self) -> bool:
        """True if baseline has been established."""
        return self._baseline is not None

    def get_baseline(self) -> Optional[SessionBaseline]:
        """
        Get the established baseline.
        Returns partial baseline if not enough windows yet
        but at least MIN_BASELINE_WINDOWS collected.
        """
        if self._baseline is not None:
            return self._baseline

        # Return partial baseline if we have enough windows
        if len(self._windows) >= MIN_BASELINE_WINDOWS:
            logger.warning(
                f"Returning partial baseline from "
                f"{len(self._windows)} windows"
            )
            self._compute_baseline()
            return self._baseline

        return None

    def force_baseline(self) -> SessionBaseline:
        """
        Force baseline computation with whatever windows we have.
        Used when session starts before enough calibration time.
        Falls back to population averages for missing data.
        """
        if len(self._windows) > 0:
            self._compute_baseline()
            return self._baseline

        logger.warning("No windows for baseline — using population defaults")
        return self._population_defaults()

    def windows_collected(self) -> int:
        return len(self._windows)

    def _compute_baseline(self):
        """
        Compute baseline from collected windows.

        Uses median instead of mean to be robust against
        outlier windows (e.g., user looked away briefly).
        """
        import statistics

        windows = self._windows

        # Eye openness baseline
        ear_values = [
            w['facial']['avg_ear']
            for w in windows
            if w['facial']['avg_ear'] > 0
        ]
        baseline_ear = (
            statistics.median(ear_values)
            if ear_values else 0.85
        )

        # Speaking rate baseline
        rate_values = [
            w['audio']['speaking_rate_wpm']
            for w in windows
            if w['audio']['speaking_rate_wpm'] > 0
        ]
        baseline_rate = (
            statistics.median(rate_values)
            if rate_values else 130.0
        )

        # Pitch variance baseline
        pitch_var_values = [
            w['audio']['pitch_variance']
            for w in windows
            if w['audio']['pitch_variance'] > 0
        ]
        baseline_pitch_var = (
            statistics.median(pitch_var_values)
            if pitch_var_values else 40.0
        )

        # Energy baseline
        energy_values = [
            w['audio']['mean_energy']
            for w in windows
            if w['audio']['mean_energy'] > 0
        ]
        baseline_energy = (
            statistics.median(energy_values)
            if energy_values else 0.02
        )

        # Brow stress baseline (their natural resting brow)
        brow_values = [
            w['facial']['brow_stress']
            for w in windows
        ]
        baseline_brow = (
            statistics.median(brow_values)
            if brow_values else 0.15
        )

        self._baseline = SessionBaseline(
            avg_ear=float(baseline_ear),
            speaking_rate_wpm=float(baseline_rate),
            pitch_variance=float(baseline_pitch_var),
            mean_energy=float(baseline_energy),
            brow_stress=float(baseline_brow),
            established=True
        )

        logger.info(
            f"Baseline established from {len(windows)} windows:\n"
            f"  EAR:          {baseline_ear:.3f}\n"
            f"  Speaking rate: {baseline_rate:.0f} wpm\n"
            f"  Pitch variance: {baseline_pitch_var:.1f}\n"
            f"  Energy:        {baseline_energy:.4f}\n"
            f"  Brow stress:   {baseline_brow:.3f}"
        )

    def _population_defaults(self) -> SessionBaseline:
        """Population average fallback values."""
        return SessionBaseline(
            avg_ear=0.85,
            speaking_rate_wpm=130.0,
            pitch_variance=40.0,
            mean_energy=0.02,
            brow_stress=0.15,
            established=False
        )
