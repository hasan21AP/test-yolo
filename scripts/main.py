import time
from datetime import datetime

import cv2

from src.config import load_config
from src.video_source import VideoSource
from src.model_runner import ModelRunner
from src.iou_tracker import IoUTracker
from src.monitor import Monitor
from src.logger import CSVLogger
from src.display import Display
from src.video_writer import VideoWriter


def main():

    # =====================================================
    # Load configuration
    # =====================================================

    config = load_config()


    # =====================================================
    # Video source
    # =====================================================

    source = VideoSource(
        config["source"]
    )

    source.open()


    # =====================================================
    # YOLO model
    # =====================================================

    model = ModelRunner(
        model_path=config["model"],
        device=config["device"],
        imgsz=config["imgsz"],
        conf=config["conf"]
    )


    # =====================================================
    # Tracker
    # =====================================================

    tracker = None

    if config["mode"] == "track":

        tracker = IoUTracker(
            iou_threshold=0.3,
            max_missed=5
        )


    # =====================================================
    # Monitoring
    # =====================================================

    monitor = Monitor(
        update_interval=5.0
    )


    # =====================================================
    # CSV Logger
    # =====================================================

    logger = None

    if config["save_csv"]:
        logger = CSVLogger()


    # =====================================================
    # Display
    # =====================================================

    display = Display()


    # =====================================================
    # Video writer
    # =====================================================

    video_writer = None

    if config["save_video"]:

        video_writer = VideoWriter(
            output_path="outputs/output.mp4",
            fps=source.fps,
            width=source.width,
            height=source.height
        )

        video_writer.open()


    # =====================================================
    # Frame counter
    # =====================================================

    frame_number = 0


    # =====================================================
    # Main loop
    # =====================================================

    try:

        while True:

            processing_start = time.perf_counter()


            # -------------------------------------------------
            # Read frame
            # -------------------------------------------------

            ret, frame = source.read()

            if not ret:
                break

            frame_number += 1


            # -------------------------------------------------
            # YOLO Detection
            # -------------------------------------------------

            model_start = time.perf_counter()

            results = model.detect(frame)

            model_end = time.perf_counter()


            model_call_latency = (
                model_end - model_start
            ) * 1000


            result = results[0]

            boxes = result.boxes


            # -------------------------------------------------
            # YOLO inference latency
            # -------------------------------------------------

            inference_latency = result.speed.get(
                "inference",
                0.0
            )


            # -------------------------------------------------
            # Convert YOLO boxes to our detections
            # -------------------------------------------------

            detections = []

            for box in boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )

                detections.append(
                    {
                        "bbox": [
                            x1,
                            y1,
                            x2,
                            y2
                        ],
                        "class_id": class_id,
                        "class_name": result.names[
                            class_id
                        ],
                        "confidence": confidence
                    }
                )


            # -------------------------------------------------
            # Tracking
            # -------------------------------------------------

            if tracker is not None:

                detections = tracker.update(
                    detections
                )

            else:

                for detection in detections:

                    detection[
                        "track_id"
                    ] = None


            detections_count = len(
                detections
            )


            # -------------------------------------------------
            # Draw normal YOLO results
            # -------------------------------------------------

            annotated_frame = result.plot()


            # -------------------------------------------------
            # Draw Track ID + Kalman estimate
            # -------------------------------------------------

            if tracker is not None:

                for detection in detections:

                    x1, y1, x2, y2 = (
                        detection["bbox"]
                    )

                    track_id = detection[
                        "track_id"
                    ]

                    predicted_x = detection[
                        "predicted_x"
                    ]

                    predicted_y = detection[
                        "predicted_y"
                    ]


                    # Track ID
                    cv2.putText(
                        annotated_frame,
                        f"ID: {track_id}",
                        (
                            int(x1),
                            max(
                                int(y1) - 25,
                                20
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )


                    # Kalman estimated position
                    cv2.circle(
                        annotated_frame,
                        (
                            int(predicted_x),
                            int(predicted_y)
                        ),
                        5,
                        (0, 0, 255),
                        -1
                    )


            # -------------------------------------------------
            # Timestamp
            # -------------------------------------------------

            timestamp = (
                datetime.now().isoformat()
            )


            # -------------------------------------------------
            # Detection CSV
            # -------------------------------------------------

            if logger is not None:

                for detection in detections:

                    x1, y1, x2, y2 = (
                        detection["bbox"]
                    )

                    width = x2 - x1
                    height = y2 - y1

                    x_center = (
                        x1 + x2
                    ) / 2

                    y_center = (
                        y1 + y2
                    ) / 2


                    logger.log_detection(
                        {
                            "frame": frame_number,
                            "timestamp": timestamp,

                            "track_id": detection["track_id"],

                            "class_id": detection["class_id"],
                            "class_name": detection["class_name"],
                            "confidence": detection["confidence"],

                            "x_center": x_center,
                            "y_center": y_center,
                            "width": width,
                            "height": height,

                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,

                            "predicted_x": detection.get(
                                "predicted_x"
                            ),

                            "predicted_y": detection.get(
                                "predicted_y"
                            ),

                            "velocity_x": detection.get(
                                "velocity_x"
                            ),

                            "velocity_y": detection.get(
                                "velocity_y"
                            )
                        }
                    )


            # -------------------------------------------------
            # Processing latency
            # -------------------------------------------------

            processing_end = time.perf_counter()

            processing_latency = (
                processing_end
                - processing_start
            ) * 1000


            # -------------------------------------------------
            # Monitoring
            # -------------------------------------------------

            updated = monitor.update(
                inference_latency=
                    inference_latency,

                model_call_latency=
                    model_call_latency,

                processing_latency=
                    processing_latency,

                detections=
                    detections_count
            )


            # -------------------------------------------------
            # Monitoring CSV
            # -------------------------------------------------

            if (
                updated
                and logger is not None
            ):

                logger.log_monitoring(
                    {
                        "timestamp":
                            timestamp,

                        "fps":
                            monitor.fps,

                        "inference_latency":
                            monitor.inference_latency,

                        "model_call_latency":
                            monitor.model_call_latency,

                        "processing_latency":
                            monitor.processing_latency,

                        "cpu_process":
                            monitor.cpu_process,

                        "cpu_system":
                            monitor.cpu_system,

                        "ram_process":
                            monitor.ram_process,

                        "ram_system":
                            monitor.ram_system,

                        "detections":
                            monitor.detections
                    }
                )


            # -------------------------------------------------
            # Draw monitoring information
            # -------------------------------------------------

            annotated_frame = (
                display.draw_monitoring(
                    annotated_frame,
                    monitor
                )
            )


            # -------------------------------------------------
            # Save output video
            # -------------------------------------------------

            if video_writer is not None:

                video_writer.write(
                    annotated_frame
                )


            # -------------------------------------------------
            # Display
            # -------------------------------------------------

            if config["show"]:

                display.show(
                    annotated_frame
                )

                if display.should_quit():
                    break


    # =====================================================
    # Cleanup
    # =====================================================

    finally:

        source.release()

        display.close()

        if logger is not None:
            logger.close()

        if video_writer is not None:
            video_writer.release()


if __name__ == "__main__":
    main()