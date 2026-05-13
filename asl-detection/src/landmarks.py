"""Hand landmark extraction using MediaPipe.

MediaPipe gives us 21 (x, y, z) landmarks per hand. We convert each frame
into a 63-dim feature vector (after normalization) that a small classifier
can map to ASL letters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import mediapipe as mp
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "mediapipe is not installed. Run: pip install -r requirements.txt"
    ) from e


NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3  # (x, y, z) per landmark


@dataclass
class HandResult:
    """One hand's landmark output for a single frame."""

    landmarks: np.ndarray  # shape (21, 3), float32, image-normalized [0, 1]
    handedness: str        # 'Left' or 'Right'
    score: float           # detection confidence in [0, 1]


class HandLandmarkExtractor:
    """Wraps MediaPipe Hands for per-frame landmark extraction.

    Usage:
        with HandLandmarkExtractor() as ext:
            res = ext.extract(rgb_frame)
            if res is not None:
                features = normalize_landmarks(res.landmarks)
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __enter__(self) -> "HandLandmarkExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._hands.close()

    def extract(self, rgb_frame: np.ndarray) -> Optional[HandResult]:
        """Run detection on an RGB image. Returns None if no hand found."""
        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            raise ValueError("rgb_frame must be HxWx3 RGB")

        result = self._hands.process(rgb_frame)
        if not result.multi_hand_landmarks:
            return None

        hand = result.multi_hand_landmarks[0]
        landmarks = np.array(
            [(lm.x, lm.y, lm.z) for lm in hand.landmark],
            dtype=np.float32,
        )
        handedness = "Right"
        score = 1.0
        if result.multi_handedness:
            classification = result.multi_handedness[0].classification[0]
            handedness = classification.label
            score = float(classification.score)

        return HandResult(landmarks=landmarks, handedness=handedness, score=score)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Translate to wrist, scale by hand size, flatten to (63,).

    Translation: subtract landmark 0 (wrist) so the hand is wrist-centered.
    Scale: divide by the max distance from the wrist to any landmark,
    making the feature invariant to how close the hand is to the camera.
    """
    if landmarks.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected shape (21, 3), got {landmarks.shape}")
    centered = landmarks - landmarks[0]
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale < 1e-8:
        scale = 1.0
    return (centered / scale).astype(np.float32).flatten()
