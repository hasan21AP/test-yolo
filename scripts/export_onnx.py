from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "yolov8n_vehicle_100e-4"
    / "weights"
    / "best.pt"
)


print(f"Loading model: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))


exported_model = model.export(
    format="onnx",
    imgsz=640,

    # Fixed 640x640 input
    dynamic=False,

    # Simplify ONNX graph
    simplify=True,

    # Keep NMS outside model
    nms=False,

    # Export using CPU
    device="cpu"
)


print()
print("ONNX export completed.")
print(f"Exported model: {exported_model}")