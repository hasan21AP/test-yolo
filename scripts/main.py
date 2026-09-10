from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


# =========================================================
# Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "runtime.yaml"


# =========================================================
# Load runtime configuration
# =========================================================
with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


MODEL_PATH = config["model"]
DEVICE = config["device"]
IMG_SIZE = config["imgsz"]
CONF = config["conf"]

MODE = config["mode"]
TRACKER = config["tracker"]

SOURCE_TYPE = config["source_type"]
SOURCE = config["source"]

SHOW = config.get("show", True)
SAVE_VIDEO = config.get("save_video", False)


# =========================================================
# Resolve model path
# =========================================================
model_path = Path(MODEL_PATH)

if not model_path.is_absolute():
    model_path = PROJECT_ROOT / model_path


# =========================================================
# Resolve source
# =========================================================
if SOURCE_TYPE == "video":
    source = Path(SOURCE)

    if not source.is_absolute():
        source = PROJECT_ROOT / source

    source = str(source)

elif SOURCE_TYPE == "webcam":
    # Example: source: 0
    source = int(SOURCE)

elif SOURCE_TYPE == "ip_camera":
    # Example:
    # source: "rtsp://192.168.1.100:8554/live"
    source = SOURCE

else:
    raise ValueError(f"Unknown source_type: {SOURCE_TYPE}")


# =========================================================
# Load YOLO model
# =========================================================
print("=" * 50)
print(f"Model       : {model_path}")
print(f"Device      : {DEVICE}")
print(f"Mode        : {MODE}")
print(f"Source type : {SOURCE_TYPE}")
print(f"Source      : {source}")

if MODE == "track":
    print(f"Tracker     : {TRACKER}")

print("=" * 50)


model = YOLO(str(model_path))


# =========================================================
# Open video / camera
# =========================================================
cap = cv2.VideoCapture(source)

if not cap.isOpened():
    raise RuntimeError(f"Could not open source: {source}")


# =========================================================
# Optional video writer
# =========================================================
writer = None

if SAVE_VIDEO:
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "output.mp4"

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    print(f"Saving video to: {output_path}")


# =========================================================
# Main loop
# =========================================================
while True:

    ret, frame = cap.read()

    if not ret:
        print("Video finished or camera disconnected.")
        break


    # -----------------------------------------------------
    # Detection only
    # -----------------------------------------------------
    if MODE == "detect":

        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONF,
            device=DEVICE,
            verbose=False
        )


    # -----------------------------------------------------
    # Detection + Tracking
    # -----------------------------------------------------
    elif MODE == "track":

        results = model.track(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONF,
            device=DEVICE,

            tracker=TRACKER,

            # Keep tracking IDs between frames
            persist=True,

            verbose=False
        )

    else:
        raise ValueError(
            f"Unknown mode: {MODE}. "
            "Use 'detect' or 'track'."
        )


    # =====================================================
    # Draw results
    # =====================================================
    annotated_frame = results[0].plot()


    # =====================================================
    # Save processed video
    # =====================================================
    if writer is not None:
        writer.write(annotated_frame)


    # =====================================================
    # Display
    # =====================================================
    if SHOW:

        cv2.imshow(
            f"YOLO - {MODE.upper()}",
            annotated_frame
        )

        key = cv2.waitKey(1) & 0xFF

        # Q or ESC to stop
        if key == ord("q") or key == 27:
            break


# =========================================================
# Cleanup
# =========================================================
cap.release()

if writer is not None:
    writer.release()

cv2.destroyAllWindows()

print("Program finished.")