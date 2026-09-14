# train_banana_detector.py
#
# Fine-tunes YOLOv8n (starting from the standard COCO-pretrained weights)
# on the Roboflow "Banana Ripening Process" dataset to produce a
# banana-specific detector.
#
# IMPORTANT - how this fits into the existing pipeline:
# This dataset has 6 classes (freshripe, freshunripe, overripe, ripe,
# rotten, unripe). We are training on all 6 so YOLO learns to draw a
# tight box around every individual banana regardless of its ripeness
# stage. When this detector is later wired into the pipeline, its
# predicted CLASS LABEL should be ignored - only its bounding boxes are
# used to crop individual bananas. Ripeness classification continues to
# come from the existing, already-verified EfficientnetBo.h5 classifier,
# exactly as before. This script does not modify or touch that
# classifier, its preprocessing, or the bunch-analysis logic.
#
# This is purely a new, additive training step. It does not change
# models/EfficientnetBo.h5, the bunch analyzer, or any existing pipeline
# code.

from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="detector_training_data/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        name="banana_detector_finetune",
    )

    print("[SUCCESS] Training complete.")
    print("Best weights saved at: runs/detect/banana_detector_finetune/weights/best.pt")
    print("Copy that file to models/banana_detector.pt to use it in the pipeline.")

if __name__ == "__main__":
    main()
