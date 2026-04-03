"""
Phase 1: Facial feature computation using MediaPipe blendshapes + landmarks.

MediaPipe 0.10.33 provides 52 blendshape scores automatically.
Head pose computed from rotation vector directly (not RQDecomp3x3)
to avoid the pitch +-180 bug on Mac webcam.
"""

import numpy as np
import cv2
import time
import logging
from typing import Optional, Tuple
from src.vision.face_tracker import FaceTracker, FaceResult
from src.contracts import FacialFeatureVector

logger = logging.getLogger(__name__)


# Standard 3D face model (mm, generic human face)
# Order: nose tip, chin, left eye corner, right eye corner,
#        left mouth corner, right mouth corner
FACE_3D_MODEL = np.array([
    [0.0,      0.0,    0.0  ],
    [0.0,    -63.6,  -12.5  ],
    [-43.3,   32.7,  -26.0  ],
    [43.3,    32.7,  -26.0  ],
    [-28.9,  -28.9,  -24.1  ],
    [28.9,   -28.9,  -24.1  ]
], dtype=np.float64)


def get_blendshape(blendshapes: list, name: str) -> float:
    """Get blendshape score by name. Returns 0.0 if not found."""
    for bs in blendshapes:
        if bs.category_name == name:
            return float(bs.score)
    return 0.0


def compute_eye_openness(blendshapes: list) -> Tuple[float, float, float]:
    """
    Eye openness from blink blendshapes.
    Inverted: 0 = closed, 1 = open.

    Returns:
        (left_openness, right_openness, avg_openness)
    """
    left_open  = 1.0 - get_blendshape(blendshapes, 'eyeBlinkLeft')
    right_open = 1.0 - get_blendshape(blendshapes, 'eyeBlinkRight')
    avg_open   = (left_open + right_open) / 2.0
    return left_open, right_open, avg_open


def compute_brow_stress(blendshapes: list) -> float:
    """
    Brow stress from brow blendshapes.
    Combines brow furrowing (down) and inner brow raise (concern).

    Returns:
        Stress score 0-1
    """
    brow_inner_up  = get_blendshape(blendshapes, 'browInnerUp')
    brow_down_left = get_blendshape(blendshapes, 'browDownLeft')
    brow_down_right = get_blendshape(blendshapes, 'browDownRight')
    brow_down_avg  = (brow_down_left + brow_down_right) / 2.0
    stress = (brow_down_avg * 0.6) + (brow_inner_up * 0.4)
    return float(min(stress, 1.0))


def compute_mouth_openness(blendshapes: list) -> float:
    """Mouth openness via jawOpen blendshape. Returns 0-1."""
    return get_blendshape(blendshapes, 'jawOpen')


def compute_duchenne_smile(blendshapes: list) -> Tuple[bool, float]:
    """
    Detect genuine (Duchenne) smile.
    Requires BOTH mouth corners lifting AND cheek squinting.

    Returns:
        (is_duchenne, confidence 0-1)
    """
    mouth_smile_avg = (
        get_blendshape(blendshapes, 'mouthSmileLeft') +
        get_blendshape(blendshapes, 'mouthSmileRight')
    ) / 2.0

    cheek_squint_avg = (
        get_blendshape(blendshapes, 'cheekSquintLeft') +
        get_blendshape(blendshapes, 'cheekSquintRight')
    ) / 2.0

    mouth_smiling  = mouth_smile_avg  > 0.15
    eyes_crinkling = cheek_squint_avg > 0.05
    is_duchenne    = mouth_smiling and eyes_crinkling

    confidence = float(np.sqrt(mouth_smile_avg * cheek_squint_avg)) \
        if is_duchenne else 0.0

    return is_duchenne, confidence


def compute_head_pose(
    landmarks: list,
    frame_shape: Tuple[int, int]
) -> Tuple[float, float, float]:
    """
    Compute head pose (pitch, yaw, roll) in degrees.

    Uses rotation vector decomposition directly instead of
    RQDecomp3x3 which causes pitch +-180 bug on Mac webcam.

    Strategy:
    1. solvePnP gives us a rotation vector
    2. Rodrigues converts to 3x3 rotation matrix
    3. We extract Euler angles with standard ZYX decomposition
       which gives stable values near 0 when looking straight

    Returns:
        (pitch, yaw, roll) in degrees
        Looking straight at camera: all three near 0
        Eye contact zone: abs(yaw) < 15, abs(pitch) < 20
    """
    h, w = frame_shape

    # Approximate camera matrix (no calibration file needed)
    focal_length = w
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array([
        [focal_length, 0.0,  cx],
        [0.0, focal_length,  cy],
        [0.0,          0.0, 1.0]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Get 2D pixel positions of 6 reference landmarks
    indices = FaceTracker.HEAD_POSE_INDICES
    face_2d = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in indices],
        dtype=np.float64
    )

    success, rvec, tvec = cv2.solvePnP(
        FACE_3D_MODEL,
        face_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    # Rotation vector → rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # ZYX Euler angle decomposition
    # rmat[i][j] notation follows standard rotation matrix layout
    #
    # pitch (X): arctan2(R32, R33)
    # yaw   (Y): arctan2(-R31, sqrt(R32^2 + R33^2))
    # roll  (Z): arctan2(R21, R11)

    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    gimbal_lock = sy < 1e-6

    if not gimbal_lock:
        pitch = np.arctan2( rmat[2, 1],  rmat[2, 2])
        yaw   = np.arctan2(-rmat[2, 0],  sy)
        roll  = np.arctan2( rmat[1, 0],  rmat[0, 0])
    else:
        pitch = np.arctan2(-rmat[1, 2],  rmat[1, 1])
        yaw   = np.arctan2(-rmat[2, 0],  sy)
        roll  = 0.0

    pitch_deg = float(np.degrees(pitch))
    yaw_deg   = float(np.degrees(yaw))
    roll_deg  = float(np.degrees(roll))

    # Mac webcam coordinate fix:
    # solvePnP returns pitch near +-180 instead of near 0
    # when looking straight. Normalize to -90/+90 range.
    if pitch_deg > 90:
        pitch_deg = pitch_deg - 180.0
    elif pitch_deg < -90:
        pitch_deg = pitch_deg + 180.0

    return pitch_deg, yaw_deg, roll_deg


def compute_eye_contact(
    avg_eye_openness: float,
    yaw: float,
    pitch: float,
    openness_threshold: float = 0.3,
    yaw_threshold: float = 12.0,
    pitch_threshold: float = 15.0
) -> bool:
    """
    True if person is looking at the camera.

    Three conditions:
    1. Eyes sufficiently open
    2. Head not turned too far left/right (yaw)
    3. Head not tilted too far up/down (pitch)
    """
    eyes_open   = avg_eye_openness > openness_threshold
    not_turned  = abs(yaw)   < yaw_threshold
    not_tilted  = abs(pitch) < pitch_threshold
    return eyes_open and not_turned and not_tilted


def extract_facial_feature_vector(
    face_result: FaceResult,
    baseline_brow_stress: Optional[float] = None
) -> FacialFeatureVector:
    """
    Master function — compute all facial features from one FaceResult.
    This is the only function other modules should call.

    Args:
        face_result: From FaceTracker.process_frame()
        baseline_brow_stress: Personal baseline from first 15s.
                              If None, raw brow stress is returned.

    Returns:
        FacialFeatureVector with all features + timestamp
    """
    if face_result is None:
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

    left_open, right_open, avg_open = compute_eye_openness(
        face_result.blendshapes
    )

    raw_brow_stress = compute_brow_stress(face_result.blendshapes)

    if baseline_brow_stress is not None and baseline_brow_stress > 0:
        brow_stress = float(min(
            max(raw_brow_stress / baseline_brow_stress - 1.0, 0.0),
            1.0
        ))
    else:
        brow_stress = raw_brow_stress

    mar = compute_mouth_openness(face_result.blendshapes)

    duchenne, _ = compute_duchenne_smile(face_result.blendshapes)

    pitch, yaw, roll = compute_head_pose(
        face_result.landmarks,
        face_result.frame_shape
    )

    eye_contact = compute_eye_contact(avg_open, yaw, pitch)

    return FacialFeatureVector(
        left_ear=left_open,
        right_ear=right_open,
        avg_ear=avg_open,
        mar=mar,
        brow_stress=brow_stress,
        head_yaw=yaw,
        head_pitch=pitch,
        head_roll=roll,
        eye_contact=eye_contact,
        duchenne_smile=duchenne,
        timestamp=time.time(),
        face_detected=True
    )
