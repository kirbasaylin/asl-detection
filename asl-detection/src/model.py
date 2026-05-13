"""TensorFlow/Keras model for ASL letter classification from hand landmarks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "tensorflow is not installed. Run: pip install -r requirements.txt"
    ) from e

from .landmarks import FEATURE_DIM


def build_model(num_classes: int, dropout: float = 0.3) -> keras.Model:
    """Small MLP — landmark-only features don't need a heavy network.

    Architecture:
        63 -> 128 (relu) -> dropout -> 64 (relu) -> dropout -> num_classes (softmax)

    Params: ~12k. Trains in seconds; inference is sub-millisecond on CPU.
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(FEATURE_DIM,), name="landmarks"),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="asl_landmark_classifier",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_labels(labels: List[str], path: Path) -> None:
    """Persist the label index so inference can map argmax → letter."""
    path.write_text(json.dumps(labels), encoding="utf-8")


def load_labels(path: Path) -> List[str]:
    return json.loads(path.read_text(encoding="utf-8"))
