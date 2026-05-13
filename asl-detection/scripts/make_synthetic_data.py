"""Generate a small synthetic feature dataset so train.py runs out of the box.

Real-world use: get a Kaggle ASL dataset and call `build_feature_dataset`.
This synthetic generator exists so anyone cloning the repo can run training
in ~10 seconds without any external data.

Usage:
    python scripts/make_synthetic_data.py --output data/features.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_class_template(rng: np.random.Generator) -> np.ndarray:
    """Create a single (21, 3) wrist-relative template for a 'letter'."""
    template = np.zeros((21, 3), dtype=np.float32)
    template[1:] = rng.normal(0, 0.3, size=(20, 3))
    template[0] = 0.0
    return template


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/features.npz"))
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--samples-per-class", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", type=float, default=0.05)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    labels = [f"CLASS_{i:02d}" for i in range(args.num_classes)]
    templates = [make_class_template(rng) for _ in range(args.num_classes)]

    X_list, y_list = [], []
    for idx, tmpl in enumerate(templates):
        for _ in range(args.samples_per_class):
            noisy = tmpl + rng.normal(0, args.noise, size=tmpl.shape).astype(np.float32)
            # Same normalization the real pipeline applies
            centered = noisy - noisy[0]
            scale = np.max(np.linalg.norm(centered, axis=1))
            if scale < 1e-8:
                scale = 1.0
            X_list.append((centered / scale).astype(np.float32).flatten())
            y_list.append(idx)

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, X=X, y=y, labels=np.array(labels))
    print(f"Wrote {args.output}: X={X.shape}, y={y.shape}, classes={len(labels)}")


if __name__ == "__main__":
    main()
