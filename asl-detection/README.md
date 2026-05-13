# ASL Detection Model

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A landmark-based American Sign Language gesture classifier. MediaPipe extracts 21 hand landmarks per frame, a small TensorFlow MLP classifies them into letters in real time. Sub-millisecond inference on CPU.

## Pipeline

```
webcam frame → MediaPipe Hands → 21 (x,y,z) landmarks
            → wrist-centered, scale-normalized 63-d vector
            → 128 → 64 → softmax(num_classes)
            → predicted letter
```

The classifier head has only ~12,000 parameters because the heavy lifting (turning pixels into a hand pose) is already done by MediaPipe.

## Quick start (Windows / macOS / Linux)

### 1. Install dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Try it with synthetic data (no dataset needed)
```bash
python scripts/make_synthetic_data.py --output data/features.npz
python scripts/train.py --features data/features.npz --output models/asl.keras --epochs 30
```

You'll see something like:
```
Final validation accuracy: 1.000
Saved model: models/asl.keras
```

### 3. Train on a real dataset
Download an ASL hand-pose dataset (e.g. [Kaggle's ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)) and arrange it as:
```
data/raw/
    A/  *.jpg
    B/  *.jpg
    ...
    Z/  *.jpg
```
Then:
```bash
python scripts/train.py --raw data/raw --output models/asl.keras --epochs 60
```

### 4. Real-time webcam demo
```bash
python scripts/predict_webcam.py --model models/asl.keras
```
Show your hand to the camera; the predicted letter appears with confidence and FPS. Press `q` to quit.

## Project structure
```
src/
    landmarks.py    MediaPipe wrapper + normalization
    model.py        Keras MLP definition
    dataset.py      Image folder → feature .npz
scripts/
    train.py                Train and evaluate
    predict_webcam.py       Live webcam inference
    make_synthetic_data.py  Generates fake features for smoke testing
```

## Why landmarks instead of raw pixels?

A landmark-based classifier is:
- **Faster** — ~12k params vs millions for a CNN over pixels
- **More robust** — invariant to skin tone, lighting, and background
- **Cheaper to train** — minutes on a CPU, not hours on a GPU
- **More interpretable** — you can directly inspect which finger positions drive a prediction

The trade-off is that anything MediaPipe can't see (e.g. occluded fingers) the classifier can't see either.

## Author

**Aylin Kirbas** — ECE & GIS @ The Ohio State University. [aylinnkirbas_@outlook.com](mailto:aylinnkirbas_@outlook.com) · [LinkedIn](https://www.linkedin.com/in/aylin-kirbas-200139267)

## License

MIT — see [LICENSE](LICENSE).
