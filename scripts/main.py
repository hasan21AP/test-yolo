import time
from datetime import datetime

from src.config import load_config
from src.video_source import VideoSource
from src.model_runner import ModelRunner
from src.monitor import Monitor
from src.logger import CSVLogger
from src.display import Display
from src.video_writer import VideoWriter


def main():

    config = load_config()

    source = VideoSource(config["source"])
    source.open()

    model = ModelRunner(
        model_path=config["model"],
        device=config["device"],
        imgsz=config["imgsz"],
        conf=config["conf"]
    )

    monitor = Monitor(update_interval=5.0)

    logger = CSVLogger()

    display = Display()

    video_writer = None

    if config["save_video"]:
        video_writer = VideoWriter(
            output_path="outputs/output.mp4",
            fps=source.fps,
            width=source.width,
            height=source.height
        )

        video_writer.open()

    frame_number = 0

    try:

        while True:

            processing_start = time.perf_counter()

            ret, frame = source.read()

            if not ret:
                break

            frame_number += 1

            model_start = time.perf_counter()

            results = model.detect(frame)

            model_end = time.perf_counter()

            model_call_latency = (
                model_end - model_start
            ) * 1000

            result = results[0]

            inference_latency = result.speed.get(
                "inference",
                0.0
            )

            boxes = result.boxes

            detections_count = len(boxes)

            annotated_frame = result.plot()

            timestamp = datetime.now().isoformat()

            if config["save_csv"]:

                for box in boxes:

                    class_id = int(box.cls[0])

                    class_name = result.names[class_id]

                    confidence = float(box.conf[0])

                    xywh = box.xywh[0].tolist()
                    xyxy = box.xyxy[0].tolist()

                    x_center, y_center, width, height = xywh
                    x1, y1, x2, y2 = xyxy

                    track_id = None

                    if box.id is not None:
                        track_id = int(box.id[0])

                    logger.log_detection(
                        {
                            "frame": frame_number,
                            "timestamp": timestamp,
                            "track_id": track_id,
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": confidence,
                            "x_center": x_center,
                            "y_center": y_center,
                            "width": width,
                            "height": height,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        }
                    )

            processing_end = time.perf_counter()

            processing_latency = (
                processing_end - processing_start
            ) * 1000

            updated = monitor.update(
                inference_latency=inference_latency,
                model_call_latency=model_call_latency,
                processing_latency=processing_latency,
                detections=detections_count
            )

            if updated and config["save_csv"]:

                logger.log_monitoring(
                    {
                        "timestamp": timestamp,
                        "fps": monitor.fps,
                        "inference_latency": monitor.inference_latency,
                        "model_call_latency": monitor.model_call_latency,
                        "processing_latency": monitor.processing_latency,
                        "cpu_process": monitor.cpu_process,
                        "cpu_system": monitor.cpu_system,
                        "ram_process": monitor.ram_process,
                        "ram_system": monitor.ram_system,
                        "detections": monitor.detections
                    }
                )

            annotated_frame = display.draw_monitoring(
                annotated_frame,
                monitor
            )

            if video_writer is not None:
                video_writer.write(annotated_frame)

            if config["show"]:

                display.show(annotated_frame)

                if display.should_quit():
                    break

    finally:

        source.release()

        logger.close()

        display.close()

        if video_writer is not None:
            video_writer.release()


if __name__ == "__main__":
    main()