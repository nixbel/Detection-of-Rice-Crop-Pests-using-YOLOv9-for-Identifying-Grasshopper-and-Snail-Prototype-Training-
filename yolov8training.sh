#!/bin/bash

# ===============================
# YOLOv8 Training Shell Script
# ===============================

# 1. Activate your conda environment
source /home/avela/miniconda3/etc/profile.d/conda.sh
conda activate /home/avela/miniconda3/envs/yolov9detection

# 2. Go to any directory where you want to save training "runs"
cd /home/avela

# 3. Install ultralytics (if not installed)
pip install --upgrade pip
pip install ultralytics

# 4. Train YOLOv8
yolo detect train \
    model=yolov8m.pt \
    data=/home/avela/miniconda3/envs/yolov9detection/yolov9/snail-grasshopper_dataset/data.yaml \
    epochs=100 \
    imgsz=640 \
    batch=8 \
    project=/home/avela/yolov8_runs \
    name=snail_grasshopper_yolov8m \
    verbose=True

# 5. Finish
echo "Training completed! Check folder: /home/avela/yolov8_runs/snail_grasshopper_yolov8m"
