import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parents[1] / "database" / "face_database.db"


def log_detection(
    source: str,
    filename: str,
    face_index: int,
    predicted_label: str,
    confidence: float,

    fake_prob: float = None,
    real_prob: float = None,
    spoof_prob: float = None,

    blur_score: float = None,
    laplacian_variance: float = None,
    fft_score: float = None,
    edge_density: float = None,

    noise_std: float = None,
    blockiness: float = None,

    saved_frame_path: str = None,
    heatmap_path: str = None,
):
    conn = sqlite3.connect(DB_PATH)

    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO detections (
            timestamp,
            source,
            filename,
            face_index,
            predicted_label,
            confidence,

            fake_prob,
            real_prob,
            spoof_prob,

            blur_score,
            laplacian_variance,
            fft_score,
            edge_density,

            noise_std,
            blockiness,

            saved_frame_path,
            heatmap_path

        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),

            source,
            filename,
            face_index,

            predicted_label,
            float(confidence),

            None if fake_prob is None else float(fake_prob),
            None if real_prob is None else float(real_prob),
            None if spoof_prob is None else float(spoof_prob),

            None if blur_score is None else float(blur_score),

            None if laplacian_variance is None else float(laplacian_variance),

            None if fft_score is None else float(fft_score),

            None if edge_density is None else float(edge_density),

            None if noise_std is None else float(noise_std),

            None if blockiness is None else float(blockiness),

            saved_frame_path,

            heatmap_path,
        ),
    )

    conn.commit()

    conn.close()