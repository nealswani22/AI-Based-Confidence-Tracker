# Real-Time Facial Confidence Tracker

An AI-powered computer vision application built with Python, Flask, OpenCV, and MediaPipe. The system processes live webcam feeds, extracts 3D facial landmarks, computes multi-metric confidence heuristics, and displays real-time visual analytics on a dark-mode web dashboard.

---

## Features

* **Real-Time Landmark Detection:** Captures 468 3D facial landmarks using MediaPipe FaceMesh.
* **Multi-Feature Confidence Scoring:**
  * **Head Pitch:** Computes upper-to-lower vertical facial geometry ratios.
  * **Mouth Curve:** Tracks corner elevation versus central top-lip alignment to measure smiles.
  * **Brow Height:** Evaluates eyebrow position relative to upper eyelids.
  * **Eye Openness:** Measures eyelid aperture for alertness detection.
* **Temporal Smoothing:** Employs a moving window average ($N=20$) to remove high-frequency metric noise.
* **Browser-Native WebRTC Architecture:** Transmits base64 canvas frames over HTTP API endpoints to prevent camera thread locking on macOS AVFoundation.

---

##  Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Backend** | Python 3.9+, Flask, OpenCV, MediaPipe, NumPy |
| **Frontend** | HTML5, CSS3, JavaScript (ES6), Bootstrap 5 |
| **Communication** | REST API (Base64 JPEG Payload Exchange) |

---

##  Scoring Heuristics

Confidence is calculated using a weighted composite score bounded between `0.00` and `1.00`:

* **Mouth Curve (35%):** Tracks smile formation.
* **Eye Openness (35%):** Measures visual engagement/eye closure.
* **Head Pitch (15%):** Evaluates posture/head tilt.
* **Brow Height (15%):** Measures eyebrow tension.

**Classification Threshold:**
* Score $\ge 0.62 \rightarrow$ **`CONFIDENT`**
* Score $< 0.62 \rightarrow$ **`UNDER-CONFIDENT`**

---

##  Project Structure

```text
Confidence_Tracker/
├── app.py                 # Flask server & MediaPipe computer vision logic
├── templates/
│   └── index.html         # Web dashboard UI & JS frame streaming pipeline
├── .gitignore             # Environment and system ignore rules
└── README.md              # Project documentation
