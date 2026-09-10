from ultralytics import YOLO
import yaml


CONFIG_PATH = "configs/yolov8n_train.yaml"


# -------------------------
# Load training config
# -------------------------
with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)


# -------------------------
# Get model
# -------------------------
model_path = config.pop("model")

print(f"Model: {model_path}")
print(f"Dataset: {config['data']}")
print(f"Epochs: {config['epochs']}")
print(f"Device: {config['device']}")
print(f"Batch size: {config['batch']}")


# -------------------------
# Load YOLO
# -------------------------
model = YOLO(model_path)


# -------------------------
# Train
# -------------------------
results = model.train(**config)


print("Training completed.")