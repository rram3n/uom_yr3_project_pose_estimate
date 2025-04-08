# Comparative Analysis of 6D Pose Estimation for Visual Servoing: CASAPose vs. YOLOv5-6D

This repository presents a comparative analysis of two modern 6D pose estimation algorithms—**CASAPose** and **YOLOv5-6D**—within the context of position-based visual servoing (PBVS). The models are evaluated on a subset of the LINEMOD dataset, with a focus on pose estimation accuracy, inference speed (FPS), and robustness to object variation and occlusion. The final aim is to assess the suitability of these models for real-time robotic manipulation.

CASAPose is a dense correspondence-based model that excels in occluded or complex visual conditions, while YOLOv5-6D is a lightweight and fast keypoint-based detector better suited for time-constrained applications. This repository contains the scripts, configuration files, and evaluation metrics used for running both models and reproducing the results.

## Implementation

To ensure environment reproducibility and dependency isolation, it is recommended to create **separate Conda environments** for each model using the instructions provided in their respective folders.

### 1. Clone the Repository

### 2. Set up Conda Envrionment (follow the instrucutions in each folder for the requirements)

### 3. Producing Inference Results
```bash
cd casapose
conda activate casapose
python test_inference_lmo_report.py
```
for YOLOv5-6D:
```bash
cd YOLOv5-6D-Pose
conda activate yolov5-6d
python detect_lmo_report.py
```
