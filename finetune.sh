#!/bin/bash

# YOLOv9 Snail and Grasshopper Detection Fine-tuning Script
# Make sure you have ultralytics installed: pip install ultralytics

# Configuration
MODEL_PATH="/home/avela/best.pt"
DATA_YAML="/home/avela/miniconda3/envs/yolov9detection/yolov9/snail-grasshopper_dataset/data.yaml"  # Path to your dataset config file
PROJECT_NAME="snail_grasshopper_detection"
EXPERIMENT_NAME="yolov9_finetune"
EPOCHS=100
BATCH_SIZE=16
IMG_SIZE=640
DEVICE=0  # GPU device (0 for first GPU, cpu for CPU)
WORKERS=8

# Training parameters
PATIENCE=50  # Early stopping patience
SAVE_PERIOD=10  # Save checkpoint every N epochs

echo "Starting YOLOv9 Fine-tuning for Snail and Grasshopper Detection..."
echo "Model: $MODEL_PATH"
echo "Dataset: $DATA_YAML"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Image Size: $IMG_SIZE"
echo ""

# Run training
yolo task=detect \
    mode=train \
    model=$MODEL_PATH \
    data=$DATA_YAML \
    epochs=$EPOCHS \
    batch=$BATCH_SIZE \
    imgsz=$IMG_SIZE \
    device=$DEVICE \
    workers=$WORKERS \
    project=$PROJECT_NAME \
    name=$EXPERIMENT_NAME \
    patience=$PATIENCE \
    save_period=$SAVE_PERIOD \
    optimizer=AdamW \
    lr0=0.001 \
    lrf=0.01 \
    momentum=0.937 \
    weight_decay=0.0005 \
    warmup_epochs=3 \
    warmup_momentum=0.8 \
    box=7.5 \
    cls=0.5 \
    dfl=1.5 \
    pose=12.0 \
    kobj=1.0 \
    label_smoothing=0.0 \
    nbs=64 \
    hsv_h=0.015 \
    hsv_s=0.7 \
    hsv_v=0.4 \
    degrees=0.0 \
    translate=0.1 \
    scale=0.5 \
    shear=0.0 \
    perspective=0.0 \
    flipud=0.0 \
    fliplr=0.5 \
    mosaic=1.0 \
    mixup=0.0 \
    copy_paste=0.0 \
    exist_ok=False \
    pretrained=True \
    verbose=True \
    seed=0 \
    deterministic=True \
    single_cls=False \
    rect=False \
    cos_lr=False \
    close_mosaic=10 \
    amp=True \
    fraction=1.0 \
    profile=False \
    freeze=None \
    multi_scale=False \
    overlap_mask=True \
    mask_ratio=4 \
    dropout=0.0 \
    val=True \
    plots=True

echo ""
echo "Training completed!"
echo "Results saved in: $PROJECT_NAME/$EXPERIMENT_NAME"
echo "Best model: $PROJECT_NAME/$EXPERIMENT_NAME/weights/best.pt"
echo "Last model: $PROJECT_NAME/$EXPERIMENT_NAME/weights/last.pt"