# Deepfake Detection System
### Hybrid Deep Learning & Computer Vision | B.Tech Final Year Project

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-orange?style=flat-square&logo=pytorch)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-RandomForest-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-lightgrey?style=flat-square)

---

## Overview

An end-to-end deepfake detection system that classifies facial media as **REAL**, **FAKE**, or **SPOOF** using a hybrid architecture combining spatial deep learning with frequency-domain analytics. The system supports image, video, and real-time webcam detection, with explainable AI outputs via Grad-CAM heatmaps.

This project was built as a B.Tech final year major project to address the growing challenge of AI-generated synthetic media.

---

## Key Highlights 

| Area | What Was Built |
|------|----------------|
| **ML Architecture** | Dual-branch hybrid model: ResNet18 CNN + Random Forest on FFT features |
| **Computer Vision** | Face detection, centroid tracking, Grad-CAM explainability |
| **Full-Stack** | Streamlit web app with image/video/webcam upload |
| **Data Engineering** | Custom preprocessing pipeline for 3 classes across image & video |
| **MLOps** | Persistent SQLite logging of predictions, confidence scores, and heatmaps |
| **Real-Time System** | Live webcam inference pipeline with per-frame classification |

---

## Technical Architecture

```
Input Media (Image / Video / Webcam)
         ↓
   Face Detection
         ↓
  ┌──────────────────────────┐
  │   Spatial Branch         │    ResNet18 CNN
  │   (Deep Learning)        │    → Facial pattern features
  └──────────┬───────────────┘
             │
  ┌──────────────────────────┐
  │   Frequency Branch       │    FFT + Laplacian + Edge Density
  │   (Classical ML)         │    → Compression & noise artifacts
  └──────────┬───────────────┘
             │
       Prediction Fusion
             ↓
   Final Label + Confidence
             ↓
  Grad-CAM Heatmap + DB Log
```

---

## Features

**Multi-Modal Detection**
Supports image uploads, video file processing, and live webcam feed with per-frame predictions.

**Hybrid ML Pipeline**
Combines a fine-tuned ResNet18 CNN for spatial feature extraction with a Random Forest classifier trained on frequency-domain signals — including FFT spectrum analysis, Laplacian blur, edge density, noise analysis, compression artifact scoring, and blockiness detection.

**Explainable AI (XAI)**
Generates Grad-CAM heatmaps to visually highlight the facial regions most influential in each prediction, making the model interpretable and audit-ready.

**Persistent Logging**
All predictions are stored in a SQLite database with prediction label, confidence score, analytics metrics, heatmap path, and timestamp for traceability.

**Interactive Web Application**
Clean Streamlit-based frontend for non-technical users to upload and analyze media with visual feedback.

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| Deep Learning | PyTorch, ResNet18 (transfer learning) |
| Classical ML | Scikit-learn, Random Forest |
| Computer Vision | OpenCV, FFT (NumPy), Grad-CAM |
| Web App | Streamlit |
| Database | SQLite |
| Language | Python 3.8+ |

---

## Project Structure

```
DEEPFAKEDETECTION/
├── analytics/              # Detection logging utilities
├── data/
│   ├── processed_faces/    # Organized real/fake/spoof training data
│   └── raw_videos/         # Raw video inputs
├── database/               # SQLite schema and setup
├── explainability/         # Grad-CAM visualization module
├── face_tracking/          # Centroid tracker for video inference
├── inference/              # Image and video inference scripts
├── models/
│   ├── saved_models/       # Trained model weights (.pth)
│   ├── spatial_cnn/        # ResNet18 training pipeline
│   └── frequency_branch/   # Random Forest training pipeline
├── realtime_system/        # Webcam detection pipeline
├── webapp/                 # Streamlit web application
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd DEEPFAKEDETECTION

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Initialize the database
python database/create_database.py
```

### Training the Models

```bash
# Train spatial CNN (ResNet18)
python models/spatial_cnn/train_spatial_model.py

# Train frequency classifier (Random Forest)
python models/frequency_branch/train_frequency_model.py
```

### Running Inference

```bash
# Test on a single image
python inference/test_on_file.py --path data/processed_faces/real/sample.jpg

# Test on a video file
python inference/test_on_file.py --path data/raw_videos/fake/fake1.mp4

# Launch real-time webcam detection (press Q to quit)
python realtime_system/realtime_pipeline.py

# Launch the web application
streamlit run webapp/app.py
```

---

## Example Output

```
Input:   Real human face
Output:  Prediction: REAL  |  Confidence: 0.94

Input:   Deepfake / phone screen
Output:  Prediction: FAKE  |  Confidence: 0.91
```

---

## What I Learned

- Designing and implementing a multi-branch ML architecture from scratch
- Applying transfer learning with ResNet18 for domain-specific classification
- Extracting and engineering frequency-domain features using FFT and OpenCV
- Building an explainable AI pipeline with Grad-CAM for model interpretability
- Integrating inference systems with a full-stack web application and persistent storage

---

## Roadmap

- [ ] Temporal LSTM branch for video sequence modeling
- [ ] Transformer-based detection (ViT)
- [ ] Face recognition integration for identity-aware analysis
- [ ] Cloud deployment (AWS / GCP)
- [ ] Mobile application support

---

## About

Developed by **[Your Name]** as a B.Tech Final Year Major Project.

**Domain:** Artificial Intelligence · Computer Vision · Media Forensics



*This project is intended for academic and research purposes.*