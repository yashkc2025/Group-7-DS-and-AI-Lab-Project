from ultralytics import YOLO
import torch

def initialize_architecture():
    print("--- Milestone 3: Model Architecture Setup ---")
    
    # Load YOLOv8l (Large) with pre-trained weights
    model = YOLO('yolov8l.pt') 
    
    # 1. Print Model Architecture Summary
    print("\nModel Summary:")
    model.info()

    # 2. Check for Hardware Acceleration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nSetup verified on device: {device}")
    
    return model

def dry_run_inference(model):
    """
    Perform a 'Dry Run' to prove the End-to-End pipeline is ready.
    In M3, we use a sample image to show the input/output flow works.
    """
    print("\n--- Performing Pipeline Verification (Dry Run) ---")
    
    results = model.predict(source="./dataset/images/train/India_1276.jpg", imgsz=640)

if __name__ == "__main__":
    yolo_large = initialize_architecture()
    dry_run_inference(yolo_large)