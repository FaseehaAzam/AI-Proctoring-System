# AI Smart Exam Monitoring & Proctoring System

**Course:** Introduction to Programming for AI  
**Group:** Hamza (SP24-BAI-016) | Hashmah Haq (SP24-BAI-017) | Faseeha Azam (SP24-BAI-012)

---

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Virtual Environment
**Windows:**
```bash
venv\Scripts\activate
```
**Mac/Linux:**
```bash
source venv/bin/activate
```

### 3. Install All Libraries
```bash
pip install -r requirements.txt
```

### 4. Run the System
```bash
python main.py
```

---

## Project Structure
```
AI_Proctoring_System/
├── main.py                    ← Entry point (run this)
├── requirements.txt           ← All libraries
├── README.md
│
├── auth/
│   ├── __init__.py
│   └── login.py               ← Student login + Regex validation
│
├── calibration/
│   ├── __init__.py
│   └── focus_game.py          ← AI eye calibration game
│
├── detection/
│   ├── __init__.py
│   ├── yolo_detector.py       ← Phone + person detection (YOLOv8)
│   ├── eye_tracker.py         ← Gaze tracking (OpenCV)
│   └── face_detector.py       ← Face presence detection
│
├── analytics/
│   ├── __init__.py
│   ├── score_calculator.py    ← Cheating Risk Score formula
│   └── regression.py          ← Linear Regression on behavior data
│
├── reports/
│   ├── __init__.py
│   └── report_generator.py    ← Graphs + PDF report
│
├── logs/
│   ├── sessions.json          ← Per-student session data
│   └── warnings.csv           ← Warning event log
│
├── models/
│   └── yolov8n.pt             ← Auto-downloaded on first run
│
└── utils/
    ├── __init__.py
    └── helpers.py             ← Shared utilities
```

---

## Features
- Student login with Regex-validated Roll Number
- 30-second eye calibration game
- Real-time phone + multi-person detection via YOLOv8
- Eye gaze tracking with OpenCV
- JSON + CSV warning logging
- Linear Regression cheating trend prediction
- Matplotlib analytics graphs
- PDF report generation with ReportLab
