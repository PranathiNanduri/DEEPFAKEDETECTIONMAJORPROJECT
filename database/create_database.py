import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "face_database.db"


def create_database():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT NOT NULL,

            source TEXT,

            filename TEXT,

            face_index INTEGER,

            predicted_label TEXT NOT NULL,

            confidence REAL NOT NULL,

            fake_prob REAL,
            real_prob REAL,
            spoof_prob REAL,

            blur_score REAL,

            laplacian_variance REAL,

            fft_score REAL,

            edge_density REAL,

            noise_std REAL,

            blockiness REAL,

            saved_frame_path TEXT,

            heatmap_path TEXT
        )
    """)

    conn.commit()
    conn.close()

    print(f"Database created successfully: {DB_PATH}")


if __name__ == "__main__":
    create_database()