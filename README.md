# AI-Powered Road Damage Detection & Classification (RDD2022)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/Model-YOLOv11l-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Executive Summary
This project implements a high-performance computer vision pipeline designed for the autonomous monitoring of Indian road infrastructure. Utilizing the state-of-the-art **YOLOv11** architecture, the system identifies and categorizes 5 types of structural distress with a focus on real-time edge deployment.

### Key Features:
* **mAP50:** 0.46 (Optimized for India-specific RDD2022 dataset).
* **State-of-the-Art Architecture:** Leverages **YOLOv11l** with C3k2/C2PSA blocks for high-precision detection.
* **Image Enhancement:** Integrated **CLAHE** preprocessing to handle the unique lighting/dust conditions of Indian roads.
* **Production API:** Scalable FastAPI implementation for real-time mobile/web integration.

---

## 🏗️ Project Architecture
The repository is structured to separate experimental research from production-ready code:

```text
├── data/               # Dataset configurations and labels
├── deployment/         # Production API (FastAPI) & Client testing
├── docs/               # Technical reports, Manuals, and PPTX
├── models/             # Serialized weights (PyTorch, ONNX, OpenVINO)
├── milestones/          # Milestone history and EDA
├── src/                # Core Modular Logic
│   ├── init_folders.py # Environment setup
│   ├── preprocess.py   # CLAHE & Spatial augmentations
│   ├── train.py        # Training & Hyperparameter orchestration
│   └── evaluate.py     # Metrics & Class-wise performance
└── outputs/            # Automated Artifact Storage
    ├── csv/            # Performance logs
    └── plots/          # Confusion Matrix, PR Curves, F1-Score
```
## 📂 Key Folders Explained
| Folder | Purpose |
| :--- | :--- |
| **`data/`** | Contains dataset configurations (`data.yaml`) and preprocessing logic. |
| **`src/`** | The "Engine" of the project; contains modular Python scripts for training and evaluation. |
| **`deployment/`** | Production-ready files including the FastAPI `server.py` and API test scripts. |
| **`models/`** | Stores final serialized weights in `.pt`, `.onnx`, and OpenVINO formats. |
| **`outputs/`** | **Auto-generated** results including performance CSVs, PR Curves, and detection samples. |
| **`milestones/`** | Archival milestone notebooks documenting the research and EDA phases. |

## 🚀 Installation & Setup
### 1. Environment Initialization
#### Clone the repository
```
git clone https://github.com/yashkc2025/Group-7-DS-and-AI-Lab-Project
cd Group-7-DS-and-AI-Lab-Project
```
#### Virtual Environment Setup

It is **strongly recommended** to use a Python virtual environment to isolate project dependencies and maintain environment stability.

#### Create Virtual Environment

```
# On Windows
python -m venv .venv

# On macOS/Linux
python3 -m venv .venv
```

#### Activate Virtual Environment

```
# On Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# On Windows (Command Prompt)
.venv\Scripts\activate.bat

# On macOS/Linux
source .venv/bin/activate
```

## Install dependencies

```
pip install -r requirements.txt
```

## Initialize project sub-directories
```
python src/init_folders.py
```

## 2. Deployment (API)
Start the inference server locally:
```
uvicorn deployment.server:app --host 0.0.0.0 --port 8000
```

## 📊 Evaluation & Metrics
The model was evaluated against a held-out test set from the RDD2022-India dataset.
* **Primary Metric:** mAP50 = 0.48
* **Optimizations:**
  * **CLAHE:** Applied to normalize lighting conditions in 180°C Indian environments.
  * **INT8 Quantization:** Achieved a 70% reduction in model size while maintaining 98.5% of FP32 accuracy.

## 👥 Team Members

| Name | Email |
| :--- | :--- |
| **Yash Kumar** | [yash.email@example.com](mailto:yash.email@example.com)
| **Neeraj Yadav** | 21f1005729@ds.study.iitm.ac.in
| **Rahul Yadav** | 22f1001680@ds.study.iitm.ac.in
| **Kisalay Pan** | [kisalay.email@example.com](mailto:kisalay.email@example.com)
| **Nishant Kumar** | [nishant.email@example.com](mailto:nishant.email@example.com)

## ⚖️ License

> This project is part of the **IITM Data Science and AI Lab** curriculum. 

All rights reserved to the authors and the Indian Institute of Technology Madras (IITM). Use of this codebase is restricted to academic evaluation and non-commercial research purposes.
