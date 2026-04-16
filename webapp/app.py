from pathlib import Path
import sys
import tempfile
import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

CANONICAL_CLASSES = ["fake", "real", "spoof"]

CUSTOM_CSS = """
<style>
.main {
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.result-box {
    padding: 18px;
    border-radius: 18px;
    background: white;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    margin-bottom: 18px;
}
.metric-box {
    padding: 14px;
    border-radius: 14px;
    background: white;
    box-shadow: 0 5px 14px rgba(0,0,0,0.06);
    text-align: center;
}
.real {
    color: #16a34a;
    font-weight: 800;
    font-size: 28px;
}
.fake {
    color: #dc2626;
    font-weight: 800;
    font-size: 28px;
}
.small-muted {
    color: #5b6470;
    font-size: 0.95rem;
}
</style>
"""


def load_project_modules():
    from inference.deepfake_inference import infer_spatial
    from inference.frequency_inference import infer_frequency
    from analytics.detection_logger import log_detection
    return infer_spatial, infer_frequency, log_detection


def probs_to_map(result):
    class_names = result.get("class_names", CANONICAL_CLASSES)
    probs = result["probs"]

    prob_map = {"fake": 0.0, "real": 0.0, "spoof": 0.0}
    for i, cls in enumerate(class_names):
        if i < len(probs) and cls in prob_map:
            prob_map[cls] = float(probs[i])
    return prob_map


def fuse_predictions(spatial_result, frequency_result):
    s = probs_to_map(spatial_result)
    f = probs_to_map(frequency_result)

    return {
        "fake": 0.6 * s["fake"] + 0.4 * f["fake"],
        "real": 0.6 * s["real"] + 0.4 * f["real"],
        "spoof": 0.6 * s["spoof"] + 0.4 * f["spoof"],
    }


# REAL vs FAKE only, ignoring spoof in final comparison
def final_decision(probs, blur_score):
    fake_p = float(probs["fake"])
    real_p = float(probs["real"])

    if real_p >= fake_p:
        return "REAL", real_p
    else:
        return "FAKE", fake_p


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


def analyze_image_bgr(image_bgr, save_heatmap_name="streamlit_image"):
    infer_spatial, infer_frequency, log_detection = load_project_modules()

    detected = detect_largest_face(image_bgr)
    if detected is None:
        raise ValueError("No clear face detected. Please upload a clearer frontal face image.")

    face_crop, face_box = detected

    spatial = infer_spatial(face_crop, save_heatmap_name=save_heatmap_name)
    frequency = infer_frequency(face_crop)

    fused_probs = fuse_predictions(spatial, frequency)
    analytics = frequency["analytics"]
    label, confidence = final_decision(fused_probs, analytics["blur_score"])

    return {
        "face_crop": face_crop,
        "face_box": face_box,
        "spatial": spatial,
        "frequency": frequency,
        "fused_probs": fused_probs,
        "analytics": analytics,
        "label": label,
        "confidence": confidence,
        "logger": log_detection,
    }


def process_video_file(video_path, sample_every_n=10):
    infer_spatial, infer_frequency, _ = load_project_modules()

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    fused_history = []
    preview_frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_every_n == 0:
            detected = detect_largest_face(frame)
            if detected is None:
                frame_idx += 1
                continue

            face_crop, (x, y, w, h) = detected

            spatial = infer_spatial(face_crop)
            frequency = infer_frequency(face_crop)
            fused_probs = fuse_predictions(spatial, frequency)
            fused_history.append(fused_probs)

            label, conf = final_decision(fused_probs, frequency["analytics"]["blur_score"])

            draw = frame.copy()
            color = (0, 255, 0) if label == "REAL" else (0, 0, 255)

            cv2.rectangle(draw, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                draw,
                f"{label} {conf:.2f}",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            if len(preview_frames) < 6:
                preview_frames.append(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB))

        frame_idx += 1

    cap.release()

    if not fused_history:
        return None

    avg_fused = {
        "fake": float(np.mean([x["fake"] for x in fused_history])),
        "real": float(np.mean([x["real"] for x in fused_history])),
        "spoof": float(np.mean([x["spoof"] for x in fused_history])),
    }

    label, confidence = final_decision(avg_fused, blur_score=100.0)

    return {
        "label": label,
        "confidence": confidence,
        "avg_fused": avg_fused,
        "preview_frames": preview_frames,
    }


def render_result_box(label, confidence):
    label_class = "real" if label == "REAL" else "fake"

    st.markdown(
        f"""
        <div class='result-box'>
            <h3>Final Decision</h3>
            <p class='{label_class}'>{label}</p>
            <p class='small-muted'>Confidence: {confidence:.4f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if label == "REAL":
        st.success("✅ Genuine face detected")
    else:
        st.error("🚨 Fake face detected")


def render_metric_cards(fake_val, real_val, spoof_val):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class='metric-box'>
                <h4>Fake</h4>
                <h3>{fake_val:.4f}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class='metric-box'>
                <h4>Real</h4>
                <h3>{real_val:.4f}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class='metric-box'>
                <h4>Spoof</h4>
                <h3>{spoof_val:.4f}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )


st.set_page_config(page_title="Deepfake Detection System", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class='result-box'>
        <h1>🛡️ Deepfake Detection System</h1>
        <p class='small-muted'>
            Upload a face image or video to classify it as REAL or FAKE and view the Grad-CAM heatmap.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.title("⚙️ Dashboard")
    st.markdown("---")
    st.info("For best results, upload clear frontal face images.")
    st.write("Modes:")
    st.write("• Image Detection")
    st.write("• Video Detection")
    st.write("• Webcam Snapshot")

tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Detection", "Webcam Snapshot"])

with tab1:
    st.subheader("Upload Image")
    img_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="img")

    if img_file is not None:
        try:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            result = analyze_image_bgr(image_bgr, save_heatmap_name=Path(img_file.name).stem)

            left, right = st.columns(2)

            with left:
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.image(
                    cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                    caption="Uploaded Image",
                    use_container_width=True
                )
                st.image(
                    cv2.cvtColor(result["face_crop"], cv2.COLOR_BGR2RGB),
                    caption="Detected Face Crop",
                    use_container_width=True
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with right:
                render_result_box(result["label"], result["confidence"])
                render_metric_cards(
                    result["fused_probs"]["fake"],
                    result["fused_probs"]["real"],
                    result["fused_probs"]["spoof"]
                )

                if result["spatial"]["heatmap_path"]:
                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    st.image(
                        result["spatial"]["heatmap_path"],
                        caption="Grad-CAM Heatmap",
                        use_container_width=True
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                with st.expander("📊 View Detailed Analytics"):
                    a = result["analytics"]
                    st.write({
                        "blur_score": round(a["blur_score"], 4),
                        "fft_score": round(a["fft_score"], 4),
                        "edge_density": round(a["edge_density"], 4),
                        "noise_std": round(a["noise_std"], 4),
                        "blockiness": round(a["blockiness"], 4),
                    })

            result["logger"](
                source="streamlit_image",
                filename=img_file.name,
                face_index=0,
                predicted_label=result["label"].lower(),
                confidence=float(result["confidence"]),
                fake_prob=float(result["fused_probs"]["fake"]),
                real_prob=float(result["fused_probs"]["real"]),
                spoof_prob=float(result["fused_probs"]["spoof"]),
                blur_score=float(result["analytics"]["blur_score"]),
                laplacian_variance=float(result["analytics"]["laplacian_variance"]),
                fft_score=float(result["analytics"]["fft_score"]),
                edge_density=float(result["analytics"]["edge_density"]),
                noise_std=float(result["analytics"]["noise_std"]),
                blockiness=float(result["analytics"]["blockiness"]),
                heatmap_path=result["spatial"]["heatmap_path"],
            )

        except Exception as e:
            st.error(f"Image analysis failed: {e}")

with tab2:
    st.subheader("Upload Video")
    vid_file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"], key="vid")
    sample_every_n = st.slider("Analyze every Nth frame", 1, 30, 10)

    if vid_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(vid_file.name).suffix) as tmp:
                tmp.write(vid_file.read())
                tmp_path = Path(tmp.name)

            with st.spinner("Analyzing video..."):
                result = process_video_file(tmp_path, sample_every_n=sample_every_n)

            if result is None:
                st.error("No usable face frames detected in the video.")
            else:
                render_result_box(result["label"], result["confidence"])
                render_metric_cards(
                    result["avg_fused"]["fake"],
                    result["avg_fused"]["real"],
                    result["avg_fused"]["spoof"]
                )

                if result["preview_frames"]:
                    st.markdown("### Preview Frames")
                    cols = st.columns(min(3, len(result["preview_frames"])))
                    for i, frame in enumerate(result["preview_frames"][:3]):
                        cols[i].image(frame, use_container_width=True)

        except Exception as e:
            st.error(f"Video analysis failed: {e}")

with tab3:
    st.subheader("Webcam Snapshot Detection")
    cam_img = st.camera_input("Capture a frame from webcam")

    if cam_img is not None:
        try:
            file_bytes = np.asarray(bytearray(cam_img.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            result = analyze_image_bgr(image_bgr, save_heatmap_name="webcam_snapshot")

            left, right = st.columns(2)

            with left:
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.image(
                    cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                    caption="Captured Frame",
                    use_container_width=True
                )
                st.image(
                    cv2.cvtColor(result["face_crop"], cv2.COLOR_BGR2RGB),
                    caption="Detected Face Crop",
                    use_container_width=True
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with right:
                render_result_box(result["label"], result["confidence"])
                render_metric_cards(
                    result["fused_probs"]["fake"],
                    result["fused_probs"]["real"],
                    result["fused_probs"]["spoof"]
                )

                if result["spatial"]["heatmap_path"]:
                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    st.image(
                        result["spatial"]["heatmap_path"],
                        caption="Grad-CAM Heatmap",
                        use_container_width=True
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                with st.expander("📊 View Detailed Analytics"):
                    a = result["analytics"]
                    st.write({
                        "blur_score": round(a["blur_score"], 4),
                        "fft_score": round(a["fft_score"], 4),
                        "edge_density": round(a["edge_density"], 4),
                        "noise_std": round(a["noise_std"], 4),
                        "blockiness": round(a["blockiness"], 4),
                    })

        except Exception as e:
            st.error(f"Webcam analysis failed: {e}")