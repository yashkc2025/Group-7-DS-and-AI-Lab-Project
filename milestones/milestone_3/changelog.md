**ADDED**

* Experimental tiered strategy using YOLOv8s (baseline), YOLOv8m (balanced), and YOLOv8l (upper-bound).
* Specific training hyperparameters including AdamW optimizer, 0.01 learning rate, and Early Stopping.
* Comprehensive evaluation framework featuring mAP@50-95 and per-class F1-scores.
* Literature-based justification for utilizing COCO pre-trained weights.
* Post-Training Quantization (PTQ) calibration details for inference optimization.

**CHANGED**

* Updated backbone specification from CSPDarknet53 to C2f modules to reflect YOLOv8 architecture.
* Revised small object detection logic to emphasize decoupled heads and anchor-free detection over layer count.
* Replaced model-size justification for class imbalance with oversampling and loss function tuning (CIoU/DFL).
* Expanded technical details for the OpenVINO inference pipeline to support sub-100ms CPU latency claims.

**FIXED**

* Corrected technical inaccuracies regarding the YOLOv8 backbone and feature fusion mechanisms.
* Resolved weak conceptual links between model parameter count and minority class detection.
