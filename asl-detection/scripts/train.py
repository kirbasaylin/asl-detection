"""Train the ASL letter classifier.

Two ways to use this script:

  1. From raw images:
       python scripts/train.py --raw data/raw --output models/asl.keras

  2. From a pre-extracted .npz (skips the slow landmark step):
       python scripts/train.py --features data/features.npz --output models/asl.keras
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import build_feature_dataset, load_feature_dataset
from src.model import build_model, save_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASL classifier")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--raw", type=Path, help="Folder of class subdirectories")
    src.add_argument("--features", type=Path, help="Pre-extracted .npz file")
    parser.add_argument(
        "--output", type=Path, default=Path("models/asl.keras"),
        help="Output .keras model path",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.raw is not None:
        feat_path = args.output.with_suffix("").parent / "features.npz"
        X, y, labels = build_feature_dataset(args.raw, feat_path)
    else:
        X, y, labels = load_feature_dataset(args.features)

    print(f"Dataset: X={X.shape}, y={y.shape}, classes={len(labels)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_frac, random_state=args.seed, stratify=y,
    )

    model = build_model(num_classes=len(labels))
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )

    model.save(args.output)
    save_labels(labels, args.output.with_suffix(".labels.json"))

    # Evaluation
    val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print("\nClassification report (validation set):")
    print(classification_report(y_val, val_pred, target_names=labels, zero_division=0))

    cm_path = args.output.with_suffix(".confusion.json")
    cm = confusion_matrix(y_val, val_pred).tolist()
    cm_path.write_text(json.dumps({"labels": labels, "matrix": cm}, indent=2))

    final_acc = history.history["val_accuracy"][-1]
    print(f"\nFinal validation accuracy: {final_acc:.3f}")
    print(f"Saved model:       {args.output}")
    print(f"Saved labels:      {args.output.with_suffix('.labels.json')}")
    print(f"Saved confusion:   {cm_path}")


if __name__ == "__main__":
    main()
