from pathlib import Path
import sys
from datetime import datetime

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from analytics.detection_logger import log_detection
from inference.deepfake_inference import infer_spatial
from inference.frequency_inference import infer_frequency
from face_tracking.centroid_tracker import CentroidTracker


DETECTION_FRAMES_DIR = PROJECT_ROOT / "data" / "detection_frames"
DETECTION_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

HEATMAP_DIR = PROJECT_ROOT / "data" / "heatmaps"
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.45
BLUR_THRESHOLD = 10
CANONICAL_CLASSES = ["fake", "real", "spoof"]


def probs_to_map(result):
    class_names = result.get("class_names", CANONICAL_CLASSES)
    probs = result["probs"]

    prob_map = {
        "fake": 0.0,
        "real": 0.0,
        "spoof": 0.0,
    }

    for i, cls in enumerate(class_names):
        if i < len(probs) and cls in prob_map:
            prob_map[cls] = float(probs[i])

    return prob_map


def fuse_predictions(spatial_result, frequency_result):
    spatial_probs = probs_to_map(spatial_result)
    freq_probs = probs_to_map(frequency_result)

    fused_probs = {
        "fake": 0.6 * spatial_probs["fake"] + 0.4 * freq_probs["fake"],
        "real": 0.6 * spatial_probs["real"] + 0.4 * freq_probs["real"],
        "spoof": 0.6 * spatial_probs["spoof"] + 0.4 * freq_probs["spoof"],
    }

    return fused_probs


def final_decision(probs, blur_score):
    fake_p = float(probs["fake"])
    real_p = float(probs["real"])
    spoof_p = float(probs["spoof"])

    suspicious_score = max(fake_p, spoof_p)

    if real_p >= 0.35:
        return "REAL", real_p

    if suspicious_score >= 0.40:
        return "FAKE", suspicious_score

    if blur_score < 8:
        return "UNCERTAIN", max(real_p, fake_p, spoof_p)

    return "UNCERTAIN", max(real_p, fake_p, spoof_p)


def main(camera_index=0):
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    tracker = CentroidTracker(max_disappeared=10)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("Press Q to Quit")
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80),
        )

        rects = []
        for (x, y, w, h) in faces:
            rects.append((x, y, x + w, y + h))

        tracker.update(rects)

        for idx, (x1, y1, x2, y2) in enumerate(rects):
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            heatmap_name = None
            if frame_count % 20 == 0:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                heatmap_name = f"heatmap_{stamp}"

            spatial = infer_spatial(face, save_heatmap_name=heatmap_name)
            frequency = infer_frequency(face)

            analytics = frequency["analytics"]
            blur_score = float(analytics["blur_score"])

            fused_probs = fuse_predictions(spatial, frequency)

            label, confidence = final_decision(fused_probs, blur_score)

            if label == "REAL":
                color = (0, 255, 0)
            elif label == "FAKE":
                color = (0, 0, 255)
            elif label == "SPOOF":
                color = (0, 165, 255)
            else:
                color = (255, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, y1 - 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
            )

            cv2.putText(
                frame,
                f"Blur:{blur_score:.1f}",
                (x1, y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"R:{fused_probs['real']:.2f} F:{fused_probs['fake']:.2f} S:{fused_probs['spoof']:.2f}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            if frame_count % 20 == 0:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                frame_path = DETECTION_FRAMES_DIR / f"frame_{stamp}.jpg"

                cv2.imwrite(str(frame_path), face)

                log_detection(
                    source="webcam",
                    filename=frame_path.name,
                    face_index=idx,
                    predicted_label=label.lower(),
                    confidence=float(confidence),
                    fake_prob=float(fused_probs["fake"]),
                    real_prob=float(fused_probs["real"]),
                    spoof_prob=float(fused_probs["spoof"]),
                    blur_score=blur_score,
                    laplacian_variance=float(analytics["laplacian_variance"]),
                    fft_score=float(analytics["fft_score"]),
                    edge_density=float(analytics["edge_density"]),
                    noise_std=float(analytics["noise_std"]),
                    blockiness=float(analytics["blockiness"]),
                    saved_frame_path=str(frame_path),
                    heatmap_path=spatial["heatmap_path"],
                )

        cv2.imshow("Advanced Deepfake Detection", frame)

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()