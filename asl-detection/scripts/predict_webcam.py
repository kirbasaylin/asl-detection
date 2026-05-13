"""Real-time ASL letter detection from a webcam.

Usage:
    python scripts/predict_webcam.py --model models/asl.keras

Press 'q' or Esc to quit.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.landmarks import HandLandmarkExtractor, normalize_landmarks
from src.model import load_labels


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_landmarks(frame, landmarks_xy):
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in landmarks_xy]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 180), 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        color = (255, 180, 60) if i == 0 else (180, 255, 255)
        cv2.circle(frame, p, 4, color, -1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mirror", action="store_true", default=True)
    args = parser.parse_args()

    model = keras.models.load_model(args.model)
    labels = load_labels(args.model.with_suffix(".labels.json"))
    print(f"Loaded model with {len(labels)} classes: {labels}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    last = time.time()
    fps = 0.0
    smoothed_probs: np.ndarray | None = None
    SMOOTH = 0.7

    with HandLandmarkExtractor(static_image_mode=False) as extractor:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = extractor.extract(rgb)
            pred_label = "—"
            confidence = 0.0
            if result is not None:
                features = normalize_landmarks(result.landmarks)[None, :]
                probs = model.predict(features, verbose=0)[0]
                smoothed_probs = (
                    probs if smoothed_probs is None
                    else SMOOTH * smoothed_probs + (1 - SMOOTH) * probs
                )
                idx = int(np.argmax(smoothed_probs))
                pred_label = labels[idx]
                confidence = float(smoothed_probs[idx])
                draw_landmarks(frame, result.landmarks)
            else:
                smoothed_probs = None

            now = time.time()
            dt = now - last
            last = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)

            txt = f"{pred_label}  {confidence*100:5.1f}%   {fps:4.1f} FPS"
            cv2.rectangle(frame, (8, 8), (380, 50), (10, 14, 10), -1)
            cv2.putText(frame, txt, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 184, 77), 2, cv2.LINE_AA)

            cv2.imshow("ASL Detection — press q to quit", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
