**Milestone 1 Submission: AI-Powered Road Damage Detection System**

# **1\. Problem Statement**

The goal of this project is to detect potholes and major cracks both longitudinal and transverse in real-time, using images captured by smartphones or dashcams mounted inside moving vehicles on Indian roads.

## **In Scope**

| Aspect | Specification |
| :---- | :---- |
| Damage Types | Potholes, Longitudinal \+ transverse cracks |
| Dataset | RDD2022 Indian subset (\~7,000 images) |
| Input | Single RGB images (640×480, smartphone quality) |
| Deployment | Offline evaluation \+ simulated edge (laptop CPU, \<100ms/frame) |
| Metrics | mAP@0.5:0.95, Precision/Recall/F1, FPS, false positives per minute |

 

## **Explicitly Out of Scope**

•       Minor cracks, alligator cracks, and construction zone damage

•       Live vehicle deployment or any hardware integration

•       Multi-modal sensors such as LiDAR, thermal, or stereo cameras

•       Training data from non-Indian roads

•       Full video processing pipelines

 

# **2\. Why This Problem Matters**

Pothole-related accidents in India rose by 53% between 2020 and 2024, now accounting for over 3,713 cases every year. India's road network spans 5.5 million kilometres, and keeping it safe through manual inspection alone costs the government around ₹500 crore annually. This is a figure that's simply not sustainable.

Our system is designed for smartphones and dashcams used in personal vehicles or civic fleet cars, making crowd-sourced road monitoring a realistic option. This is especially relevant in monsoon-heavy India, where poor road conditions are linked to roughly 70% of road accidents.

# **3\. Current State-of-the-Art Solutions**

We reviewed over 25 papers published between 2018 and 2026\. YOLO-based architectures dominate real-time detection tasks, and the RDD2022 dataset has become the standard benchmark for road damage research.

## **Key Approaches & Benchmarks**

 

| Category | Representative Works | Datasets | Performance | Limitations |
| :---- | :---- | :---- | :---- | :---- |
| YOLO Real-Time | YOLOv5/v8, SEA-YOLOv8 | RDD2022 | mAP@0.5: 0.55-0.63 (India); F1: 0.70-0.77; FPS: 20-50 GPU | Domain shift to India (−15% mAP), slow on CPU |
| Advanced Detection | YOLO-RD, RT-DETR | RDD2022 India/Japan | mAP: 0.65-0.75; Precision: \>0.75 | High compute, poor on small potholes |
| Segmentation | U-Net, DeepLab | Crack500, custom | IoU: 0.70-0.75 | \<10 FPS, misses irregular shapes |
| Lightweight | MobileNet+YOLO | Custom India | FPS: 15-25 CPU; Recall: 0.72 | Lower accuracy (mAP: 0.58) |

 

The official RDD2022 baseline uses YOLOv5s, achieving mAP@0.5 \= 0.62 overall, but this drops to 0.55 on the Indian subset, largely due to differences in lighting conditions and road textures.

## **Standard Metrics & Success Criteria**

•       Primary: mAP@0.5:0.95 (target \> 0.70)

•       Supporting: Precision \= TP/(TP+FP), Recall \= TP/(TP+FN), F1 \= 2PR/(P+R)

•       Practical: FPS \> 10 on CPU, fewer than 5 false positives per minute, minimum detection distance

# **4\. Shortcomings & Research Gaps**

## **Critical Limitations**

•       Domain Bias: Models trained on Japanese or US roads perform poorly on Indian roads, with a 15-25% drop in mAP. Monsoon-related wetness, shadow patterns, and asphalt textures are largely unrepresented in existing training data.

•       Edge Deployment: Most published solutions rely on GPU hardware

•       Small Object Detection: Potholes smaller than 100px are frequently missed, particularly when there's motion blur or partial occlusion.

•       India-Specific Gap: No existing work specifically fine-tunes on the RDD2022 Indian subset, which contains over 7,000 images.

 

## **The Opportunity**

By combining India-specific fine-tuning with targeted augmentation (simulating blur, shadows, and rain) and model distillation, we expect a 10-20% mAP improvement over baseline while maintaining 15+ FPS on a standard laptop CPU. This directly addresses the lack of affordable, practical tools for civic road monitoring.

# **5\. Our Contribution Dimensions**

 

| Contribution Area | How We Address It | Expected Impact |
| :---- | :---- | :---- |
| Performance | India-specific fine-tuning | \+15% mAP vs. baseline |
| Efficiency | Model distillation | 15+ FPS on laptop CPU |
| New Context | Smartphone deployment | Practical civic use case |
| Robustness | Cross-region generalisation | Works on unseen Indian roads |

 

To our knowledge, this is the first undergraduate project to combine India-specific fine-tuning, edge optimisation, and cross-region testing on the RDD2022 dataset.

# **6\. Evidence Plan \- Proving Success**

## **Quantitative Validation**

•       A mAP/F1 comparison table contrasting the baseline YOLOv8 against our fine-tuned model (80/20 train/test split)

•       An ablation study examining the effect of augmentation and the trade-off between model size and inference speed

## **Qualitative Validation**

•       Detection videos with bounding box overlays

•       Error analysis covering false positive and false negative case studies

•       Confusion matrix visualisation

