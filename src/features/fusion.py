"""
Phase 3: Temporal alignment and fusion of facial + audio streams.

Takes a list of FacialFeatureVectors collected over a 2-second window
and one AudioFeatureVector for the same window.

Averages the facial features across all frames in the window to produce
one representative facial summary. Merges with audio into a FusedWindow.

Design decision: We average facial features rather than taking the last
frame because individual frames can be noisy (blinks, head movements).
The 2-second average is more stable and representative.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import time
import logging
import numpy as np
from typing import List, Optional
from src.contracts import (
    FacialFeatureVector,
    AudioFeatureVector,
    FusedWindow
)

logger = logging.getLogger(__name__)


def average_facial_vectors(
    facial_vectors: List[FacialFeatureVector]
) -> FacialFeatureVector:
    """
    Average a list of FacialFeatureVectors into one representative vector.

    Only averages vectors where face_detected=True.
    If no faces detected in the window, returns a zero vector.

    Args:
        facial_vectors: List of FacialFeatureVectors from a 2s window

    Returns:
        Single averaged FacialFeatureVector
    """
    # Filter to frames where face was actually detected
    detected = [f for f in facial_vectors if f.get('face_detected', False)]

    if not detected:
        logger.debug("No faces detected in window — returning zero vector")
        return FacialFeatureVector(
            left_ear=0.0,
            right_ear=0.0,
            avg_ear=0.0,
            mar=0.0,
            brow_stress=0.0,
            head_yaw=0.0,
            head_pitch=0.0,
            head_roll=0.0,
            eye_contact=False,
            duchenne_smile=False,
            timestamp=time.time(),
            face_detected=False
        )

    n = len(detected)

    # Average all float fields
    avg_left_ear    = sum(f['left_ear']    for f in detected) / n
    avg_right_ear   = sum(f['right_ear']   for f in detected) / n
    avg_ear         = sum(f['avg_ear']     for f in detected) / n
    avg_mar         = sum(f['mar']         for f in detected) / n
    avg_brow_stress = sum(f['brow_stress'] for f in detected) / n
    avg_head_yaw    = sum(f['head_yaw']    for f in detected) / n
    avg_head_pitch  = sum(f['head_pitch']  for f in detected) / n
    avg_head_roll   = sum(f['head_roll']   for f in detected) / n

    # Eye contact: percentage of frames with eye contact
    # Stored as float (0-1) representing fraction of frames
    eye_contact_pct = sum(
        1 for f in detected if f['eye_contact']
    ) / n

    # Eye contact boolean: True if contact in >50% of frames
    eye_contact = eye_contact_pct > 0.5

    # Duchenne smile: True if detected in >30% of frames
    # (genuine smiles are sustained, not just one frame)
    duchenne_pct   = sum(
        1 for f in detected if f['duchenne_smile']
    ) / n
    duchenne_smile = duchenne_pct > 0.3

    return FacialFeatureVector(
        left_ear=float(avg_left_ear),
        right_ear=float(avg_right_ear),
        avg_ear=float(avg_ear),
        mar=float(avg_mar),
        brow_stress=float(avg_brow_stress),
        head_yaw=float(avg_head_yaw),
        head_pitch=float(avg_head_pitch),
        head_roll=float(avg_head_roll),
        eye_contact=eye_contact,
        duchenne_smile=duchenne_smile,
        timestamp=time.time(),
        face_detected=True
    )


def create_fused_window(
    facial_vectors: List[FacialFeatureVector],
    audio_vector: AudioFeatureVector,
    window_start: float,
    window_end: float
) -> FusedWindow:
    """
    Create a FusedWindow from facial vectors and audio vector.

    This is the main function called by pipeline.py every 2 seconds.

    Args:
        facial_vectors: All FacialFeatureVectors collected in this window
                        Typically 15-30 vectors at 15fps over 2 seconds
        audio_vector:   AudioFeatureVector for the same 2-second window
        window_start:   Unix timestamp when window started
        window_end:     Unix timestamp when window ended

    Returns:
        FusedWindow ready for baseline and classification
    """
    averaged_facial = average_facial_vectors(facial_vectors)

    return FusedWindow(
        facial=averaged_facial,
        audio=audio_vector,
        window_start=window_start,
        window_end=window_end
    )


def get_eye_contact_percentage(
    facial_vectors: List[FacialFeatureVector]
) -> float:
    """
    Compute exact eye contact percentage from raw frame list.

    More precise than the boolean in averaged_facial because it
    uses all raw frames rather than the averaged vector.

    Returns:
        Float 0-1 representing fraction of frames with eye contact
    """
    detected = [f for f in facial_vectors if f.get('face_detected', False)]
    if not detected:
        return 0.0
    return sum(1 for f in detected if f['eye_contact']) / len(detected)
