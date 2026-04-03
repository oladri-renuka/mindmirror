"""
Phase 1: MediaPipe FaceLandmarker wrapper (Tasks API for MediaPipe 0.10.30+).

Works on Apple Silicon Mac with MediaPipe 0.10.33.
Uses the new Tasks API with FaceLandmarker model file.

Key upgrade over legacy API:
- 52 blendshape scores computed automatically
- These replace manual EAR/MAR/brow computation
- Head pose still computed manually from landmarks in facial_features.py
"""

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import cv2
import numpy as np
import logging
import os
from typing import Optional, Tuple, NamedTuple

logger = logging.getLogger(__name__)


class FaceResult(NamedTuple):
    """
    Clean container for FaceLandmarker output.
    Passed to facial_features.py for feature computation.
    """
    landmarks: list          # 478 landmark objects, each with .x .y .z
    blendshapes: list        # 52 blendshape objects, each with .category_name .score
    transformation_matrix: Optional[np.ndarray]  # 4x4 matrix for head pose
    frame_shape: Tuple[int, int]  # (height, width) for pixel conversion


class FaceTracker:
    """
    Wraps MediaPipe FaceLandmarker (Tasks API).

    Usage:
        tracker = FaceTracker()

        result = tracker.process_frame(frame)
        if result is not None:
            # result.landmarks  -> 478 points
            # result.blendshapes -> 52 scores
        
        debug_frame = tracker.draw_landmarks(frame, result)
        tracker.close()
    """

    # Landmark indices used by facial_features.py
    # Kept here as single source of truth

    # Head pose reference points (6 points for solvePnP)
    HEAD_POSE_INDICES = [
        1,    # Nose tip
        152,  # Chin
        226,  # Left eye left corner
        446,  # Right eye right corner
        57,   # Left mouth corner
        287   # Right mouth corner
    ]

    # Eye indices (still useful for pixel-space eye contact check)
    LEFT_EYE_CENTER = 33
    RIGHT_EYE_CENTER = 362

    # Blendshape names we use in facial_features.py
    # These are the category_name values from result.face_blendshapes
    BLENDSHAPE_EYE_BLINK_LEFT = 'eyeBlinkLeft'
    BLENDSHAPE_EYE_BLINK_RIGHT = 'eyeBlinkRight'
    BLENDSHAPE_BROW_INNER_UP = 'browInnerUp'
    BLENDSHAPE_BROW_DOWN_LEFT = 'browDownLeft'
    BLENDSHAPE_BROW_DOWN_RIGHT = 'browDownRight'
    BLENDSHAPE_MOUTH_SMILE_LEFT = 'mouthSmileLeft'
    BLENDSHAPE_MOUTH_SMILE_RIGHT = 'mouthSmileRight'
    BLENDSHAPE_CHEEK_SQUINT_LEFT = 'cheekSquintLeft'
    BLENDSHAPE_CHEEK_SQUINT_RIGHT = 'cheekSquintRight'
    BLENDSHAPE_JAW_OPEN = 'jawOpen'
    BLENDSHAPE_MOUTH_PUCKER = 'mouthPucker'
    BLENDSHAPE_BROW_OUTER_UP_LEFT = 'browOuterUpLeft'
    BLENDSHAPE_BROW_OUTER_UP_RIGHT = 'browOuterUpRight'

    def __init__(
        self,
        model_path: str = "models/face_landmarker.task",
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Args:
            model_path: Path to face_landmarker.task model file
            min_face_detection_confidence: Initial detection threshold
            min_face_presence_confidence: Presence threshold per frame
            min_tracking_confidence: Tracking continuity threshold
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Download with:\n"
                f"curl -o {model_path} -L https://storage.googleapis.com/"
                f"mediapipe-models/face_landmarker/face_landmarker/float16/1/"
                f"face_landmarker.task"
            )

        options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=mp_vision.RunningMode.IMAGE
        )

        self._detector = mp_vision.FaceLandmarker.create_from_options(options)

        # Statistics
        self._frames_processed = 0
        self._faces_detected = 0

        logger.info("FaceTracker initialized (Tasks API, 478 landmarks + 52 blendshapes)")

    def process_frame(self, frame: np.ndarray) -> Optional[FaceResult]:
        """
        Extract landmarks and blendshapes from a single BGR frame.

        Args:
            frame: BGR numpy array from OpenCV

        Returns:
            FaceResult if face detected, None otherwise.
            Always check for None before using result.
        """
        if frame is None:
            return None

        self._frames_processed += 1

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap in MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Run detection
        result = self._detector.detect(mp_image)

        if not result.face_landmarks:
            return None

        self._faces_detected += 1

        # Extract transformation matrix if available
        transform_matrix = None
        if result.facial_transformation_matrixes:
            transform_matrix = np.array(
                result.facial_transformation_matrixes[0].data
            ).reshape(4, 4)

        return FaceResult(
            landmarks=result.face_landmarks[0],
            blendshapes=result.face_blendshapes[0] if result.face_blendshapes else [],
            transformation_matrix=transform_matrix,
            frame_shape=(frame.shape[0], frame.shape[1])
        )

    def get_blendshape_score(
        self,
        blendshapes: list,
        name: str
    ) -> float:
        """
        Get score for a specific blendshape by name.

        Args:
            blendshapes: From FaceResult.blendshapes
            name: Category name e.g. 'eyeBlinkLeft'

        Returns:
            Score 0.0-1.0, or 0.0 if not found
        """
        for bs in blendshapes:
            if bs.category_name == name:
                return bs.score
        return 0.0

    def get_all_blendshapes_dict(self, blendshapes: list) -> dict:
        """
        Convert blendshape list to a name->score dictionary.
        Useful for debugging and logging.
        """
        return {bs.category_name: bs.score for bs in blendshapes}

    def get_landmark_pixels(
        self,
        landmarks: list,
        index: int,
        frame_shape: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Convert normalized landmark to pixel coordinates.

        Args:
            landmarks: From FaceResult.landmarks
            index: Landmark index (0-477)
            frame_shape: (height, width)

        Returns:
            (x_pixels, y_pixels) as integers
        """
        h, w = frame_shape
        lm = landmarks[index]
        return int(lm.x * w), int(lm.y * h)

    def draw_landmarks(
        self,
        frame: np.ndarray,
        face_result: Optional[FaceResult],
        draw_key_points: bool = True
    ) -> np.ndarray:
        """
        Draw landmarks on frame for visualization.

        Args:
            frame: BGR frame
            face_result: From process_frame(), or None
            draw_key_points: Draw the key points used for features

        Returns:
            New annotated frame (original unchanged)
        """
        annotated = frame.copy()

        if face_result is None:
            cv2.putText(
                annotated,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            return annotated

        h, w = face_result.frame_shape

        if draw_key_points:
            # Draw eye landmarks in green
            eye_indices = [33, 133, 362, 263, 159, 145, 386, 374]
            for idx in eye_indices:
                x, y = self.get_landmark_pixels(
                    face_result.landmarks, idx, face_result.frame_shape
                )
                cv2.circle(annotated, (x, y), 2, (0, 255, 0), -1)

            # Draw brow landmarks in yellow
            for idx in [107, 336, 66, 296]:
                x, y = self.get_landmark_pixels(
                    face_result.landmarks, idx, face_result.frame_shape
                )
                cv2.circle(annotated, (x, y), 3, (0, 255, 255), -1)

            # Draw nose tip in blue
            x, y = self.get_landmark_pixels(
                face_result.landmarks, 1, face_result.frame_shape
            )
            cv2.circle(annotated, (x, y), 4, (255, 0, 0), -1)

            # Draw mouth corners in orange
            for idx in [61, 291]:
                x, y = self.get_landmark_pixels(
                    face_result.landmarks, idx, face_result.frame_shape
                )
                cv2.circle(annotated, (x, y), 3, (0, 165, 255), -1)

        # Show detection rate
        cv2.putText(
            annotated,
            f"Detection: {self.get_detection_rate():.0%}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        return annotated

    def get_detection_rate(self) -> float:
        """Fraction of frames where face was detected."""
        if self._frames_processed == 0:
            return 0.0
        return self._faces_detected / self._frames_processed

    def get_stats(self) -> dict:
        return {
            "frames_processed": self._frames_processed,
            "faces_detected": self._faces_detected,
            "detection_rate": round(self.get_detection_rate(), 3)
        }

    def close(self):
        """Release resources. Always call when done."""
        self._detector.close()
        logger.info(
            f"FaceTracker closed. "
            f"Detection rate: {self.get_detection_rate():.1%}"
        )
