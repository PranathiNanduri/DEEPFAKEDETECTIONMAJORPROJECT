# DEEPFAKEDETECTIONMAJORPROJECT
# DEEPFAKEDETECTIONMAJORPROJECT
# 🛡️ Deepfake Detection System using Hybrid Deep Learning and Computer Vision

A robust AI-powered Deepfake Detection System built using **Deep Learning**, **Computer Vision**, and **Frequency Domain Analytics** to classify facial media as:

- **REAL**
- **FAKE**
- **SPOOF**

The system supports:

- **Image Detection**
- **Video Detection**
- **Real-Time Webcam Detection**
- **Grad-CAM Heatmap Visualization**
- **Database Logging**
- **Interactive Web Application**

---

# 🚀 Features

### ✅ Hybrid Detection Architecture
Combines:

- Spatial CNN (Deep Learning Face Classification)
- Frequency Domain Analysis (FFT / Noise / Compression Artifacts)
- Computer Vision Analytics

---

### ✅ Multi-Modal Input Support

Detect deepfakes from:

- Uploaded Images
- Uploaded Videos
- Live Webcam Feed

---

### ✅ Explainable AI

Generates:

- **Grad-CAM Heatmaps**
- Model Attention Visualization

---

### ✅ Database Logging

Stores detection records including:

- Prediction Label
- Confidence Score
- Analytics Metrics
- Heatmap Path
- Timestamp

---

### ✅ Real-Time Detection Pipeline

Supports:

- Live webcam facial analysis
- Real/Fake/Spoof prediction
- Heatmap saving during runtime

---


# 🧠 Machine Learning Algorithms Used

## Spatial Branch

Uses:

- **ResNet18 CNN**
- Feature Learning from Facial Spatial Patterns

---

## Frequency Branch

Uses:

- **Random Forest Classifier**
- Frequency Domain Feature Extraction

Features:

- FFT Spectrum Analysis
- Laplacian Blur Detection
- Edge Density
- Noise Analysis
- Compression Artifact Detection
- Blockiness Detection

---

# 📊 Detection Pipeline

```text
Input Media
   ↓
Face Detection
   ↓
Spatial CNN Inference
   ↓
Frequency Analytics Extraction
   ↓
Prediction Fusion Layer
   ↓
Final Classification
   ↓
Heatmap + Logging
```

---

# 🔥 Installation Guide

## Clone Repository

```bash
git clone <your-repo-link>
cd DEEPFAKEDETECTION
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Activate:

### Windows:

```bash
venv\Scripts\activate
```

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

# ⚙️ Database Setup

```bash
python database/create_database.py
```

---

# 🏋️ Training Models

## Train Spatial CNN

```bash
python models/spatial_cnn/train_spatial_model.py
```

---

## Train Frequency Model

```bash
python models/frequency_branch/train_frequency_model.py
```

---

# 🧪 Testing

## Test on Image

```bash
python inference/test_on_file.py --path data/processed_faces/real/sample.jpg
```

---

## Test on Video

```bash
python inference/test_on_file.py --path data/raw_videos/fake/fake1.mp4
```

---

# 🎥 Real-Time Webcam Detection

```bash
python realtime_system/realtime_pipeline.py
```

Press:

```bash
Q → Quit Webcam
```

---

# 🌐 Run Web Application

```bash
streamlit run webapp/app.py
```

---

# 📈 Example Output

### Input:

- Real Human Face

### Output:

```text
Prediction: REAL
Confidence: 0.94
```

---

### Input:

- Phone Screen / Deepfake Face

### Output:

```text
Prediction: FAKE
Confidence: 0.91
```

---

# 📌 Future Enhancements

- Temporal LSTM Branch for Video Sequence Analysis
- Transformer-Based Detection
- Face Recognition Integration
- Cloud Deployment
- Mobile Application Support

---

# 👨‍💻 Author

**Your Name**

Major Project — B.Tech Final Year

---

# 📜 License

This project is for educational and research purposes.
