"""Dataset utilities.

Expected dataset layout (compatible with most Kaggle ASL hand-pose datasets):

    data/raw/
        A/
            img_001.jpg
            img_002.jpg
            ...
        B/
            ...
        ...
        Z/
            ...

`build_feature_dataset` walks this tree, extracts hand landmarks from each
image with MediaPipe, normalizes them, and saves a single `.npz` file with
arrays `X` (N, 63), `y` (N,), and `labels` (list of class strings).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .landmarks import HandLandmarkExtractor, normalize_landmarks


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _discover_classes(raw_dir: Path) -> List[str]:
    classes = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())
    if not classes:
        raise FileNotFoundError(
            f"No class subdirectories under {raw_dir}. Expected raw_dir/<class>/<image>."
        )
    return classes


def build_feature_dataset(
    raw_dir: Path,
    output_path: Path,
    min_per_class: int = 5,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract features for every image and save a single .npz."""
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = _discover_classes(raw_dir)
    label_to_idx = {name: i for i, name in enumerate(labels)}

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    skipped = 0

    with HandLandmarkExtractor(static_image_mode=True) as extractor:
        for class_name in labels:
            class_dir = raw_dir / class_name
            images = [
                p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
            ]
            for img_path in tqdm(images, desc=class_name, leave=False):
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    skipped += 1
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result = extractor.extract(rgb)
                if result is None:
                    skipped += 1
                    continue
                X_list.append(normalize_landmarks(result.landmarks))
                y_list.append(label_to_idx[class_name])

    # Filter rare classes (MediaPipe can fail on stylized images)
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int32)
    keep = np.zeros_like(y, dtype=bool)
    for idx in range(len(labels)):
        mask = y == idx
        if mask.sum() >= min_per_class:
            keep |= mask
    X, y = X[keep], y[keep]

    np.savez(output_path, X=X, y=y, labels=np.array(labels))
    print(
        f"Saved {output_path} | samples={len(X)} | classes={len(labels)} | "
        f"skipped={skipped}"
    )
    return X, y, labels


def load_feature_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    data = np.load(path, allow_pickle=False)
    return data["X"], data["y"], list(data["labels"])
