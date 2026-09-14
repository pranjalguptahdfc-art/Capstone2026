from roboflow import Roboflow

rf = Roboflow(api_key="3jyyIuq4G0fmOiCUC3BL")
project = rf.workspace("fruit-ripening").project("banana-ripening-process")
version = project.version(2)

dataset = version.download("yolov8", location="detector_training_data")

print("[SUCCESS] Dataset downloaded to:", dataset.location)
print("Check detector_training_data/data.yaml to confirm class names and paths.")
