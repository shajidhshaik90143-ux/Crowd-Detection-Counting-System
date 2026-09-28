# Crowd Detection & Counting System

A strong, resume-ready computer-vision project for detecting and counting people in images, videos, and live webcam feeds.

## Features

- YOLO person detection
- Real-time people counting
- Image analysis
- Video file analysis
- Webcam mode
- Confidence and IoU controls
- Bounding boxes and labels
- Density classification
- Visual center-point density analysis
- Frame-by-frame analytics CSV
- Annotated video export
- Streamlit dashboard
- Clean modular project structure
- Automatic YOLO model download on first run

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLO
- NumPy
- Pandas
- Streamlit

## Project Structure

```text
Crowd_Detection_Counting_System/
├── app.py
├── requirements.txt
├── requirements-python311.txt
├── run.bat
├── run.sh
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   ├── __init__.py
│   └── detector.py
├── assets/
└── outputs/
```

## Recommended Environment

For Windows, use **Python 3.11.x** for the smoothest compatibility with the computer-vision stack.

> Python 3.14 may not have compatible wheels for every ML dependency on every machine. If installation errors occur on 3.14, install Python 3.11 and create a fresh virtual environment.

## Windows Setup

Open PowerShell inside the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit, normally:

```text
http://localhost:8501
```

You can also double-click `run.bat`.

## First Run

The first run downloads the selected YOLO model automatically.

Default model:

```text
yolo11n.pt
```

This is a small model intended for speed. `yolo11s.pt` is available from the sidebar when more accuracy is desired.

## How To Use

### Image

1. Select **Image**.
2. Upload JPG/PNG/WEBP.
3. The application detects people.
4. View count, density, confidence and detection coordinates.

### Video

1. Select **Video File**.
2. Upload MP4/AVI/MOV/MKV/WEBM.
3. Wait for processing.
4. Download the annotated video and CSV analytics.

### Webcam

1. Select **Webcam**.
2. Click **Start Webcam**.
3. Allow camera access.
4. The live people count appears over the camera feed.

## Density Logic

The project uses a practical screen-level heuristic based on detected people and image area:

- Empty
- Low
- Medium
- High
- Very High

These are **not official crowd-safety thresholds**. For a real deployment, calibrate the thresholds for the camera position, field of view and target environment.

## Resume Description

**Crowd Detection & Counting System — Python, OpenCV, YOLO, Streamlit**

Developed a computer-vision application that detects and counts people from images, video streams and webcam input using YOLO and OpenCV. Implemented confidence/IoU controls, density classification, annotated video generation and frame-level CSV analytics through an interactive Streamlit dashboard.

## Suggested Resume Skills

- Python
- Computer Vision
- Object Detection
- YOLO
- OpenCV
- Streamlit
- NumPy
- Pandas
- Video Analytics

## Future Enhancements

- Multi-object tracking with persistent IDs
- Line-crossing entry/exit counter
- Region-of-interest counting
- Crowd heatmap using historical detections
- Email/SMS alerts for configured thresholds
- Database storage
- REST API
- Docker deployment
- CCTV/IP camera integration
- GPU acceleration
- Dashboard charts for hourly/daily traffic

## Privacy

Only process camera/video data when you have permission to do so. Avoid collecting or storing personally identifying information unless necessary and legally permitted.
