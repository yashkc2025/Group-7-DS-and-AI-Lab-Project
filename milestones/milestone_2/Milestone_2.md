# Milestone 2 

## 1\. Dataset Overview

The dataset used for this project is the **India split** of the RDD2022 Dataset. It contains **7,706 high-resolution RGB images** in `.jpg` format, each paired with an XML annotation file stored in a separate directory.

The dataset covers four categories of road damage:

| Code | Damage Type |
| :---- | :---- |
| D00 | Longitudinal Cracks |
| D10 | Transverse Cracks |
| D20 | Alligator Cracks (mesh/spalling) |
| D40 | Potholes |

## 2\. Technical Stack

The following libraries and tools were used throughout this milestone:

- **OpenCV (cv2)** \- image loading and processing  
- **Albumentations** \- data augmentation  
- **pandas & numpy** \- data handling and analysis  
- **matplotlib & tqdm** \- visualisation and progress tracking

## 3\. Dataset Validation

Before beginning any analysis, the dataset was checked for integrity. All 7,706 images were confirmed to have a matching XML annotation file, giving a perfect 1-to-1 pairing. No missing files, corrupted images, or filename mismatches were found. The dataset was considered clean and ready for use.

## 4\. Exploratory Data Analysis (EDA)

### 4.1 Image Resolution

A sample of the first 1,000 images was analysed to understand the resolution distribution. The most common resolution was **720×720 pixels**, with only minor variation across the dataset. Resizing all images to **640×640 pixels** \- the standard input size for YOLO models \- was deemed feasible without significant distortion.

### 4.2 Visual Inspection

Random samples were visualised with bounding boxes overlaid to better understand the data. Several observations were made:

- Damage appears under varied real-world conditions, including shadows, wet and dry surfaces, and different lighting environments.  
- Multiple instances of damage often appear within a single image.  
- The four damage classes are visually distinguishable in most cases, which is encouraging for model training.

### 4.3 Class Distribution and Imbalance

A total of approximately **9,531 annotated bounding boxes** were counted across all 7,706 images. The distribution across damage classes is as follows:

| Damage Type | Class | Count | Percentage |
| :---- | :---- | :---- | :---- |
| Longitudinal Crack | D00 | 1,555 | 16.3% |
| Transverse Crack | D10 | 68 | 0.7% |
| Alligator Crack | D20 | 2,021 | 21.2% |
| Pothole | D40 | 3,187 | 33.4% |

The data shows a significant class imbalance. Potholes (D40) make up the largest share of annotations, while transverse cracks (D10) account for less than 1% of all instances with only 68 examples. If left unaddressed, this imbalance could cause the model to become biased toward predicting potholes and largely ignore rarer damage types like D10.

## 5\. Preprocessing and Augmentation Pipeline

### Augmentation Strategy

To address the class imbalance identified in the EDA, an augmentation pipeline was designed using the **Albumentations** library. The pipeline is configured to preserve bounding box coordinates in YOLO's normalised `xywh` format throughout all transformations.

The planned augmentation techniques include:

- **Geometric transforms:** Random rotation (±15–30°), horizontal flipping, scaling, and shifting  
- **Colour and lighting adjustments:** Brightness, contrast, hue, and saturation variations  
- **Weather simulation:** Mild fog, rain, and blur effects to improve model robustness  
- **Class-aware augmentation:** D10 instances receive 5–10× more augmentation passes; D00 and D20 receive 2–3× more, directly compensating for their lower representation in the dataset

## 6\. Team Consent

| Name | Roll No. | Sign |
| :---- | :---- | :---- |
| Yash Kumar | 22F3000472 | Y.K |
| Nishant Kumar | 22f3003042 | N.K |
| Rahul Yadav | 22F1001680 | R.Y |
| Neeraj Yadav | 21F1005729 | N.J |
| Kisalay Pan  | 22F2001094  | K.P |

