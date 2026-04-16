from pathlib import Path
import sys
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from inference.deepfake_inference import infer_spatial
from inference.frequency_inference import infer_frequency
from analytics.detection_logger import log_detection

CANONICAL_CLASS_ORDER = ["fake", "real", "spoof"]


def probs_to_map(result):
    class_names = result.get("class_names", CANONICAL_CLASS_ORDER)
    probs = result["probs"]

    prob_map = {name: 0.0 for name in CANONICAL_CLASS_ORDER}

    for i, name in enumerate(class_names):
        if i < len(probs) and name in prob_map:
            prob_map[name] = float(probs[i])

    return prob_map


def fuse_predictions(spatial_result, frequency_result):
    spatial_map = probs_to_map(spatial_result)
    frequency_map = probs_to_map(frequency_result)

    fused = {
        "fake": 0.6 * spatial_map["fake"] + 0.4 * frequency_map["fake"],
        "real": 0.6 * spatial_map["real"] + 0.4 * frequency_map["real"],
        "spoof": 0.6 * spatial_map["spoof"] + 0.4 * frequency_map["spoof"],
    }
    return fused


def final_decision(probs, blur_score):
    fake_p = float(probs["fake"])
    real_p = float(probs["real"])
    spoof_p = float(probs["spoof"])

    suspicious_score = max(fake_p, spoof_p)

    # Strong fake/spoof wins first
    if suspicious_score >= 0.45 and suspicious_score > real_p:
        return "FAKE", suspicious_score

    # Then real if strongest enough
    if real_p >= 0.35 and real_p > suspicious_score:
        return "REAL", real_p

    # Blur fallback
    if blur_score < 8:
        return "UNCERTAIN", max(real_p, fake_p, spoof_p)

    return "UNCERTAIN", max(real_p, fake_p, spoof_p)

def print_prob_block(title, result):
    prob_map = probs_to_map(result)
    print(f"{title:<10}: label={result['label']:<9} conf={result['confidence']:.4f}")
    print(
        f"           probs -> "
        f"fake={prob_map['fake']:.4f}, "
        f"real={prob_map['real']:.4f}, "
        f"spoof={prob_map['spoof']:.4f}"
    )


def detect_largest_face(image):
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    largest = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest

    pad = int(0.15 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image.shape[1], x + w + pad)
    y2 = min(image.shape[0], y + h + pad)

    return image[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def process_image(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read {image_path}")

    detected = detect_largest_face(image)

    if detected is None:
        print("No face detected, using full image for inference.")
        face_crop = image
        x, y, w, h = 0, 0, image.shape[1], image.shape[0]
    else:
        face_crop, (x, y, w, h) = detected

    spatial = infer_spatial(face_crop, save_heatmap_name=image_path.stem)
    frequency = infer_frequency(face_crop)

    fused_probs = fuse_predictions(spatial, frequency)
    analytics = frequency["analytics"]

    final_label, final_conf = final_decision(
        fused_probs,
        analytics["blur_score"]
    )

    log_detection(
        source="file",
        filename=image_path.name,
        face_index=0,
        predicted_label=final_label.lower(),
        confidence=float(final_conf),
        fake_prob=float(fused_probs["fake"]),
        real_prob=float(fused_probs["real"]),
        spoof_prob=float(fused_probs["spoof"]),
        blur_score=float(analytics["blur_score"]),
        laplacian_variance=float(analytics["laplacian_variance"]),
        fft_score=float(analytics["fft_score"]),
        edge_density=float(analytics["edge_density"]),
        noise_std=float(analytics["noise_std"]),
        blockiness=float(analytics["blockiness"]),
        heatmap_path=spatial["heatmap_path"],
    )

    print("\n=== FINAL RESULT ===")
    print(f"Image      : {image_path.name}")
    print(f"Face Box   : x={x}, y={y}, w={w}, h={h}")
    print_prob_block("Spatial", spatial)
    print_prob_block("Frequency", frequency)

    print(
        f"Fused      : "
        f"fake={fused_probs['fake']:.4f}, "
        f"real={fused_probs['real']:.4f}, "
        f"spoof={fused_probs['spoof']:.4f}"
    )
    print(f"Decision   : {final_label} ({final_conf:.4f})")

    print("--- Analytics ---")
    for k, v in analytics.items():
        if isinstance(v, (int, float, np.floating)):
            print(f"{k}: {float(v):.4f}")
        else:
            print(f"{k}: {v}")

    if spatial["heatmap_path"]:
        print(f"Heatmap saved: {spatial['heatmap_path']}")


def process_video(video_path: Path, sample_every_n=15):
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    fused_history = []

    print("Press q to stop video analysis window.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        display_frame = frame.copy()

        if frame_idx % sample_every_n == 0:
            detected = detect_largest_face(frame)

            if detected is not None:
                face_crop, (x, y, w, h) = detected
            else:
                face_crop = frame
                x, y, w, h = 0, 0, frame.shape[1], frame.shape[0]

            spatial = infer_spatial(face_crop)
            frequency = infer_frequency(face_crop)
            fused_probs = fuse_predictions(spatial, frequency)
            fused_history.append(fused_probs)

            final_label, final_conf = final_decision(
                fused_probs,
                frequency["analytics"]["blur_score"]
            )

            if final_label == "REAL":
                color = (0, 255, 0)
            elif final_label == "FAKE":
                color = (0, 0, 255)
            else:
                color = (255, 255, 0)

            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

            cv2.putText(
                display_frame,
                f"{final_label} {final_conf:.2f}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            cv2.putText(
                display_frame,
                f"R:{fused_probs['real']:.2f} F:{fused_probs['fake']:.2f} S:{fused_probs['spoof']:.2f}",
                (x, min(y + h + 20, display_frame.shape[0] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        cv2.imshow("Video Analysis", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    if not fused_history:
        print("No usable frames processed.")
        return

    avg_fused = {
        "fake": float(np.mean([x["fake"] for x in fused_history])),
        "real": float(np.mean([x["real"] for x in fused_history])),
        "spoof": float(np.mean([x["spoof"] for x in fused_history])),
    }

    final_label, final_conf = final_decision(avg_fused, blur_score=100.0)

    print("\n=== VIDEO RESULT ===")
    print(f"Video      : {video_path.name}")
    print(
        f"Averaged   : "
        f"fake={avg_fused['fake']:.4f}, "
        f"real={avg_fused['real']:.4f}, "
        f"spoof={avg_fused['spoof']:.4f}"
    )
    print(f"Decision   : {final_label} ({final_conf:.4f})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Image or video path")
    parser.add_argument("--sample_every_n", type=int, default=15)
    args = parser.parse_args()

    p = Path(args.path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        process_image(p)
    else:
        process_video(p, sample_every_n=args.sample_every_n)