from pathlib import Path
import cv2
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "saved_models" / "frequency_model.pth"

DEFAULT_CLASS_NAMES = ["fake", "real", "spoof"]
CANONICAL_ORDER = ["fake", "real", "spoof"]


def extract_frequency_features(image_bgr: np.ndarray):
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image provided to extract_frequency_features")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift) + 1.0)

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    low = float(mag[dist <= 20].mean())
    mid = float(mag[(dist > 20) & (dist <= 60)].mean())
    high = float(mag[dist > 60].mean())

    high_low_ratio = float(high / (low + 1e-8))

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    noise_std = float(np.std(gray.astype(np.float32) - blur.astype(np.float32)))

    vals = []
    for i in range(8, gray.shape[0], 8):
        vals.append(
            np.mean(
                np.abs(
                    gray[i, :].astype(np.float32) -
                    gray[i - 1, :].astype(np.float32)
                )
            )
        )

    for j in range(8, gray.shape[1], 8):
        vals.append(
            np.mean(
                np.abs(
                    gray[:, j].astype(np.float32) -
                    gray[:, j - 1].astype(np.float32)
                )
            )
        )

    blockiness = float(np.mean(vals) / 255.0) if vals else 0.0

    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 255.0)

    features = np.array([
        brightness,
        contrast,
        lap_var,
        edge_density,
        low,
        mid,
        high,
        high_low_ratio,
        noise_std,
        blockiness,
    ], dtype=np.float32)

    analytics = {
        "brightness": brightness,
        "contrast": contrast,
        "laplacian_variance": lap_var,
        "edge_density": edge_density,
        "fft_score": high_low_ratio,
        "low_freq_energy": low,
        "mid_freq_energy": mid,
        "high_freq_energy": high,
        "noise_std": noise_std,
        "blockiness": blockiness,
        "blur_score": lap_var,
    }

    return features, analytics


def load_frequency_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Frequency model not found at: {MODEL_PATH}")

    checkpoint = joblib.load(MODEL_PATH)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model = checkpoint["model"]
        class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    else:
        model = checkpoint
        class_names = DEFAULT_CLASS_NAMES

    return model, class_names


def align_probabilities(raw_probs, class_names):
    prob_map = {name: 0.0 for name in CANONICAL_ORDER}

    for i, cls_name in enumerate(class_names):
        if i < len(raw_probs) and cls_name in prob_map:
            prob_map[cls_name] = float(raw_probs[i])

    ordered_probs = [prob_map[name] for name in CANONICAL_ORDER]
    return ordered_probs


def infer_frequency(face_bgr: np.ndarray):
    if face_bgr is None or face_bgr.size == 0:
        raise ValueError("Empty face image provided to infer_frequency")

    model, class_names = load_frequency_model()
    feats, analytics = extract_frequency_features(face_bgr)

    raw_probs = model.predict_proba([feats])[0]
    probs = align_probabilities(raw_probs, class_names)

    pred_idx = int(np.argmax(probs))
    pred_label = CANONICAL_ORDER[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "label": pred_label,
        "confidence": confidence,
        "probs": probs,
        "class_names": CANONICAL_ORDER,
        "analytics": analytics,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise ValueError(f"Could not read {args.image}")

    result = infer_frequency(img)

    print("\n=== Frequency Inference Result ===")
    print(f"Label      : {result['label']}")
    print(f"Confidence : {result['confidence']:.4f}")
    print(f"Probabilities ({result['class_names']}): {result['probs']}")
    print("Analytics:")
    for k, v in result["analytics"].items():
        print(f"{k}: {v}")