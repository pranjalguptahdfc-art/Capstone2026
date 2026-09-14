from ultralytics import YOLO

model = YOLO(r"D:\Pranjal Files\Capstone2026\Banana-Ripeness-Bunch\models\banana_detector.pt")
print(model.names)