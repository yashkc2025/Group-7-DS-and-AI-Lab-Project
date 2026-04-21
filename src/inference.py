import os
import sys
import cv2
from ultralytics import YOLO

def run_prediction(image_source, weights_path='models/best.pt'):
    """
    Runs inference on a single image or a folder of images.
    :param image_source: Path to an image or directory of images.
    :param weights_path: Path to the trained .pt weights.
    """
    # 1. Load Model
    if not os.path.exists(weights_path):
        print(f"❌ Error: Model weights not found at {weights_path}")
        return

    model = YOLO(weights_path)
    print(f"🧠 Model loaded from {weights_path}")

    # 2. Run Inference
    # stream=True handles memory efficiently for large folders
    results = model.predict(
        source=image_source,
        conf=0.25,        # Confidence threshold
        save=False,       # We will handle saving manually for custom naming
        imgsz=640,
        device=0          # Run on GPU 0
    )

    # 3. Process and Save Results
    output_dir = 'outputs/samples'
    os.makedirs(output_dir, exist_ok=True)

    print(f"🔍 Processing detections for: {image_source}")
    
    for i, r in enumerate(results):
        # Plot the detections on the original image
        annotated_frame = r.plot()
        
        # Extract original filename or create one
        if hasattr(r, 'path'):
            base_name = os.path.basename(r.path)
        else:
            base_name = f"detection_{i}.jpg"
            
        save_path = os.path.join(output_dir, f"pred_{base_name}")
        
        # Save using OpenCV
        cv2.imwrite(save_path, annotated_frame)
        print(f"✅ Saved detection to: {save_path}")

if __name__ == "__main__":
    # Check if a path was provided via command line
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        # Default fallback for testing
        target = 'data/raw' 
        
    run_prediction(target)
