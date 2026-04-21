from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
import uvicorn
import io
import os
from PIL import Image
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="Road Damage Detection API",
    description="REST API for real-time identification of road distress using YOLOv11l",
    version="1.0.0"
)

# Global model variable (loaded on startup)
MODEL_PATH = "../models/best.pt"
model = None

@app.on_event("startup")
def load_model():
    """Load the model into memory when the server starts."""
    global model
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"⚠️ Warning: Model not found at {MODEL_PATH}. Ensure weights are moved to /models.")

@app.get("/")
def health_check():
    """Simple endpoint to verify the server is running."""
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image and returns JSON detections.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")

    # 1. Read image from the request
    try:
        request_object_content = await file.read()
        img = Image.open(io.BytesIO(request_object_content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # 2. Run Inference
    results = model.predict(img, conf=0.25, imgsz=640)
    
    # 3. Parse Detections
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class_name": model.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "bbox": [round(x, 2) for x in box.xyxy[0].tolist()] # [x1, y1, x2, y2]
            })

    return {
        "filename": file.filename,
        "detection_count": len(detections),
        "results": detections
    }

if __name__ == "__main__":
    # Run the server: uvicorn server:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
