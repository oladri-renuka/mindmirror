"""
Shared data contracts for MindMirror.
All modules import TypedDicts from here.
Never import from individual module files for type definitions.
"""

from typing import TypedDict, List, Optional


class FacialFeatureVector(TypedDict):
    # Eye openness (via blendshapes, 0=closed 1=open)
    left_ear: float
    right_ear: float
    avg_ear: float

    # Mouth openness (via jawOpen blendshape, 0=closed 1=open)
    mar: float

    # Brow stress (0=relaxed 1=stressed)
    brow_stress: float

    # Head pose in degrees
    head_yaw: float       # left(-) / right(+), eye contact if abs < 15
    head_pitch: float     # down(-) / up(+), eye contact if abs < 15
    head_roll: float      # tilt

    # Derived booleans
    eye_contact: bool     # True if looking at camera
    duchenne_smile: bool  # True if genuine smile

    # Metadata
    timestamp: float
    face_detected: bool   # False if no face in frame


class AudioFeatureVector(TypedDict):
    # Pitch
    mean_pitch: float
    pitch_variance: float
    uptalk: bool

    # Rate
    speaking_rate_wpm: float

    # Energy
    mean_energy: float
    energy_variance: float
    trailing_off: bool

    # Pauses
    boundary_pauses: int
    mid_phrase_pauses: int

    # Language
    filler_count: int
    fillers_detected: List[str]
    hedge_count: int
    hedges_detected: List[str]

    # Transcript
    transcript: str

    # Metadata
    timestamp: float


class FusedWindow(TypedDict):
    facial: FacialFeatureVector
    audio: AudioFeatureVector
    window_start: float
    window_end: float


class BehavioralWindow(TypedDict):
    fused: FusedWindow
    state: str
    confidence: float
    evidence: dict
    delta: str
    session_stats: dict
    timestamp: float


class SessionBaseline(TypedDict):
    avg_ear: float
    speaking_rate_wpm: float
    pitch_variance: float
    mean_energy: float
    brow_stress: float
    established: bool
