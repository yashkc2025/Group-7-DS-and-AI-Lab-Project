# AI-Powered Road Damage Detection & Classification (RDD2022)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/Model-YOLOv11l-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Executive Summary
This project implements a high-performance computer vision pipeline designed for the autonomous monitoring of Indian road infrastructure. Utilizing the state-of-the-art **YOLOv11** architecture, the system identifies and categorizes 5 types of structural distress with a focus on real-time edge deployment.

### Key Highlights:
* **mAP50:** 0.542 (Optimized for India-specific RDD2022 dataset).
* **Hardware Agnostic:** Optimized for both Dual T4 GPUs (Training) and CPU-based Edge devices (Inference).
* **Production Ready:** Full FastAPI integration for mobile-client communication.

---

## 🏗️ Project Architecture
The repository is structured to separate experimental research from production-ready code:

```text
├── data/               # Dataset configurations and labels
├── deployment/         # Production API (FastAPI) & Client testing
├── docs/               # Technical reports, Manuals, and PPTX
├── models/             # Serialized weights (PyTorch, ONNX, OpenVINO)
├── notebooks/          # Milestone history and EDA
├── src/                # Core Modular Logic
│   ├── init_folders.py # Environment setup
│   ├── preprocess.py   # CLAHE & Spatial augmentations
│   ├── train.py        # Training & Hyperparameter orchestration
│   └── evaluate.py     # Metrics & Class-wise performance
└── outputs/            # Automated Artifact Storage
    ├── csv/            # Performance logs
    └── plots/          # Confusion Matrix, PR Curves, F1-Score
## 🚀 Installation & Setup
### 1. Environment Initialization
# Clone the repository
git clone [https://github.com/your-username/road-damage-detection.git](https://github.com/your-username/road-damage-detection.git)
cd road-damage-detection

# Install dependencies
pip install -r requirements.txt

# Initialize project sub-directories
python src/init_folders.py
### 2. Deployment (API)
Start the inference server locally:
uvicorn deployment.server:app --host 0.0.0.0 --port 8000
## 📊 Evaluation & Metrics
The model was evaluated against a held-out test set from the RDD2022-India dataset.
Primary Metric: mAP50 = 0.542
Optimizations: * CLAHE: Applied to normalize lighting conditions in 180°C Indian environments.
INT8 Quantization: Achieved a 70% reduction in model size while maintaining 98.5% of FP32 accuracy.
