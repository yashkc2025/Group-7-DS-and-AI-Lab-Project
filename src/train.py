import os
from ultralytics import YOLO
import torch

def train_pipeline():
    # 1. Environment Setup
    # Clear cache to prevent memory fragmentation on Kaggle T4s
    torch.cuda.empty_cache()
    
    # Path Configuration
    DATA_YAML = os.path.abspath('data/data.yaml')
    OUTPUT_DIR = os.path.abspath('outputs/weights')
    
    # 2. Stage 1: YOLOv11l @ 640px (The Foundation)
    print("🚀 Starting Stage 1: Training YOLOv11l at 640px...")
    
    model = YOLO('yolo11l.pt') # Load pretrained large model
    
    model.train(
        data=DATA_YAML,
        epochs=150,           # Max epochs; patience will fire early
        imgsz=640,
        batch=16,             # Optimized for T4 VRAM
        device=0,             # Forced single-GPU for stability at the hospital
        patience=25,          # Sweet spot for convergence
        project=OUTPUT_DIR,
        name='v11l_640_stage1',
        exist_ok=True,
        # Loss Gains adjusted for Indian Road textures
        box=7.5,
        cls=1.0,
        cls_pw=1.0,           # Must be <= 1.0 to avoid AssertionError
        copy_paste=0.1        # Minority class augmentation
    )

    print("✅ Stage 1 Complete. Check 'outputs/weights/v11l_640_stage1' for results.")

if __name__ == "__main__":
    train_pipeline()
