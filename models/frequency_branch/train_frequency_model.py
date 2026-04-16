from pathlib import Path
import cv2
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed_faces"

SAVE_DIR = PROJECT_ROOT / "models" / "saved_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = SAVE_DIR / "frequency_model.pth"


CLASS_MAP = {
    "fake": 0,
    "real": 1,
    "spoof": 2
}


# ====================================
# FEATURE EXTRACTION
# ====================================
def extract_frequency_features(image_bgr: np.ndarray):

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (256, 256))

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    edges = cv2.Canny(gray, 80, 160)

    edge_density = np.mean(edges > 0)

    f = np.fft.fft2(gray)

    fshift = np.fft.fftshift(f)

    mag = np.log(np.abs(fshift) + 1.0)

    h, w = mag.shape

    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]

    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    low = mag[dist <= 20].mean()

    mid = mag[(dist > 20) & (dist <= 60)].mean()

    high = mag[dist > 60].mean()

    high_low_ratio = float(high / (low + 1e-8))

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    noise_std = float(
        np.std(
            gray.astype(np.float32) -
            blur.astype(np.float32)
        )
    )

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

    return np.array([
        float(gray.mean() / 255.0),
        float(gray.std() / 255.0),
        float(lap_var),
        float(edge_density),
        float(low),
        float(mid),
        float(high),
        high_low_ratio,
        noise_std,
        blockiness,
    ], dtype=np.float32)


# ====================================
# LOAD DATA
# ====================================
def load_samples():

    X = []
    y = []

    for label_name, label_id in CLASS_MAP.items():

        class_dir = DATA_DIR / label_name

        if not class_dir.exists():
            continue

        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):

            for p in class_dir.glob(ext):

                image = cv2.imread(str(p))

                if image is None:
                    continue

                feats = extract_frequency_features(image)

                X.append(feats)

                y.append(label_id)

    if len(X) < 10:

        raise RuntimeError(
            "Need more images to train frequency model"
        )

    return np.array(X), np.array(y)


# ====================================
# TRAIN
# ====================================
def main():

    X, y = load_samples()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=4,
        random_state=42
    )

    clf.fit(X_train, y_train)

    preds = clf.predict(X_val)

    acc = accuracy_score(y_val, preds)

    print(f"\nValidation Accuracy: {acc:.4f}")

    print(
        classification_report(
            y_val,
            preds,
            target_names=list(CLASS_MAP.keys())
        )
    )

    joblib.dump({
        "model": clf,
        "class_names": list(CLASS_MAP.keys())
    }, MODEL_PATH)

    print(f"\nSaved frequency model to {MODEL_PATH}")


if __name__ == "__main__":

    main()