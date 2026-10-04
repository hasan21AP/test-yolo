import time
from datetime import datetime

import cv2

from src.config import load_config
from src.video_source import VideoSource
from src.model_runner import ModelRunner

from src.iou_tracker import IoUTracker
from src.single_iou_tracker import SingleIoUTracker
from src.ocsort_tracker import OCSortTracker
from src.csrt_tracker import CSRTTracker

from src.monitor import Monitor
from src.logger import CSVLogger
from src.display import Display
from src.video_writer import VideoWriter


def select_highest_confidence(detections):

    if not detections:
        return None

    return max(
        detections,
        key=lambda detection: detection["confidence"]
    )


def main():

    # =====================================================
    # Load configuration
    # =====================================================

    config = load_config()

    mode = config["mode"]
    tracker_name = config.get("tracker", "single_iou")

    detection_mode = config["detection_mode"]
    detection_interval = config["detection_interval"]


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


    if mode == "track":

        # -------------------------------------------------
        # Single target IoU
        # -------------------------------------------------

        if tracker_name == "single_iou":

            tracker = SingleIoUTracker(
                iou_threshold=0.25,
                max_lost_frames=10
            )


        # -------------------------------------------------
        # Custom IoU + Kalman
        # -------------------------------------------------

        elif tracker_name == "iou":

            tracker = IoUTracker(
                iou_threshold=0.10,
                max_missed=12
            )


        # -------------------------------------------------
        # OC-SORT
        # -------------------------------------------------

        elif tracker_name == "ocsort":

            if detection_mode != "continuous":

                raise ValueError(
                    "OC-SORT currently requires "
                    "detection_mode: continuous"
                )

            tracker = OCSortTracker(
                model
            )


        # -------------------------------------------------
        # CSRT single-object tracker
        # -------------------------------------------------

        elif tracker_name == "csrt":

            tracker = CSRTTracker()


        else:

            raise ValueError(
                f"Unknown tracker: {tracker_name}"
            )


    # =====================================================
    # Monitoring
    # =====================================================

    monitor = Monitor(
        update_interval=5.0
    )


    # =====================================================
    # Logger
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
    # Runtime variables
    # =====================================================

    frame_number = 0

    last_detection_time = None

    # CSRT keeps the class information from the YOLO
    # detection used to initialize the single target.
    csrt_class_id = None
    csrt_class_name = "object"


    # =====================================================
    # Main loop
    # =====================================================

    try:

        while True:

            processing_start = (
                time.perf_counter()
            )


            # -------------------------------------------------
            # Read frame
            # -------------------------------------------------

            ret, frame = source.read()

            if not ret:
                break


            frame_number += 1


            # -------------------------------------------------
            # Default values
            # -------------------------------------------------

            detections = []

            model_call_latency = 0.0
            inference_latency = 0.0


            # =================================================
            # CSRT
            # =================================================

            if (
                mode == "track"
                and tracker_name == "csrt"
            ):

                # ---------------------------------------------
                # First target: YOLO detection + CSRT init
                # ---------------------------------------------

                if not tracker.initialized:

                    model_start = (
                        time.perf_counter()
                    )


                    results = model.detect(
                        frame
                    )


                    model_end = (
                        time.perf_counter()
                    )


                    model_call_latency = (
                        model_end
                        - model_start
                    ) * 1000


                    result = results[0]
                    boxes = result.boxes


                    inference_latency = (
                        result.speed.get(
                            "inference",
                            0.0
                        )
                    )


                    raw_detections = []


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


                        raw_detections.append(
                            {
                                "bbox": [
                                    x1,
                                    y1,
                                    x2,
                                    y2
                                ],

                                "class_id":
                                    class_id,

                                "class_name":
                                    result.names[
                                        class_id
                                    ],

                                "confidence":
                                    confidence
                            }
                        )


                    target = (
                        select_highest_confidence(
                            raw_detections
                        )
                    )


                    if target is not None:

                        tracker.init(
                            frame,
                            target["bbox"]
                        )

                        csrt_class_id = (
                            target["class_id"]
                        )

                        csrt_class_name = (
                            target["class_name"]
                        )

                        detections = [
                            {
                                "bbox":
                                    target["bbox"],

                                "track_id":
                                    1,

                                "class_id":
                                    csrt_class_id,

                                "class_name":
                                    csrt_class_name,

                                # This confidence is from the
                                # YOLO initialization frame.
                                "confidence":
                                    target["confidence"],

                                "predicted_x":
                                    None,

                                "predicted_y":
                                    None,

                                "velocity_x":
                                    None,

                                "velocity_y":
                                    None
                            }
                        ]


                # ---------------------------------------------
                # Following frames: CSRT only, no YOLO
                # ---------------------------------------------

                else:

                    tracker_start = (
                        time.perf_counter()
                    )


                    success, bbox = (
                        tracker.update(
                            frame
                        )
                    )


                    tracker_end = (
                        time.perf_counter()
                    )


                    model_call_latency = (
                        tracker_end
                        - tracker_start
                    ) * 1000


                    if success:

                        detections = [
                            {
                                "bbox":
                                    bbox,

                                "track_id":
                                    1,

                                "class_id":
                                    csrt_class_id,

                                "class_name":
                                    csrt_class_name,

                                # CSRT does not provide a YOLO-
                                # style confidence score.
                                "confidence":
                                    None,

                                "predicted_x":
                                    None,

                                "predicted_y":
                                    None,

                                "velocity_x":
                                    None,

                                "velocity_y":
                                    None
                            }
                        ]


            # =================================================
            # OC-SORT
            # =================================================

            elif (
                mode == "track"
                and tracker_name == "ocsort"
            ):

                model_start = (
                    time.perf_counter()
                )


                results = tracker.update(
                    frame
                )


                model_end = (
                    time.perf_counter()
                )


                model_call_latency = (
                    model_end
                    - model_start
                ) * 1000


                result = results[0]


                inference_latency = (
                    result.speed.get(
                        "inference",
                        0.0
                    )
                )


                boxes = result.boxes


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


                    track_id = None

                    if box.id is not None:

                        track_id = int(
                            box.id[0]
                        )


                    detections.append(
                        {
                            "bbox": [
                                x1,
                                y1,
                                x2,
                                y2
                            ],

                            "track_id":
                                track_id,

                            "class_id":
                                class_id,

                            "class_name":
                                result.names[
                                    class_id
                                ],

                            "confidence":
                                confidence,

                            "predicted_x":
                                None,

                            "predicted_y":
                                None,

                            "velocity_x":
                                None,

                            "velocity_y":
                                None
                        }
                    )


            # =================================================
            # Detect / Custom trackers
            # =================================================

            else:

                current_time = (
                    time.perf_counter()
                )


                # ---------------------------------------------
                # Decide if YOLO should run
                # ---------------------------------------------

                if detection_mode == "continuous":

                    run_yolo = True


                elif detection_mode == "interval":

                    run_yolo = (
                        last_detection_time is None
                        or
                        current_time
                        - last_detection_time
                        >= detection_interval
                    )


                else:

                    raise ValueError(
                        f"Unknown detection_mode: "
                        f"{detection_mode}"
                    )


                # =============================================
                # YOLO ON
                # =============================================

                if run_yolo:

                    model_start = (
                        time.perf_counter()
                    )


                    results = model.detect(
                        frame
                    )


                    model_end = (
                        time.perf_counter()
                    )


                    model_call_latency = (
                        model_end
                        - model_start
                    ) * 1000


                    last_detection_time = (
                        time.perf_counter()
                    )


                    result = results[0]

                    boxes = result.boxes


                    inference_latency = (
                        result.speed.get(
                            "inference",
                            0.0
                        )
                    )


                    raw_detections = []


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


                        raw_detections.append(
                            {
                                "bbox": [
                                    x1,
                                    y1,
                                    x2,
                                    y2
                                ],

                                "class_id":
                                    class_id,

                                "class_name":
                                    result.names[
                                        class_id
                                    ],

                                "confidence":
                                    confidence
                            }
                        )


                    # -----------------------------------------
                    # Tracking
                    # -----------------------------------------

                    if tracker is not None:

                        detections = (
                            tracker.update(
                                raw_detections
                            )
                        )


                    # -----------------------------------------
                    # Detection only
                    # -----------------------------------------

                    else:

                        detections = (
                            raw_detections
                        )


                        for detection in detections:

                            detection["track_id"] = None

                            detection["predicted_x"] = None
                            detection["predicted_y"] = None

                            detection["velocity_x"] = None
                            detection["velocity_y"] = None


                # =============================================
                # YOLO OFF
                # =============================================

                else:

                    if tracker is not None:

                        detections = (
                            tracker.predict_only()
                        )

                    else:

                        detections = []


            # =================================================
            # Detection count
            # =================================================

            detections_count = len(
                detections
            )


            # =================================================
            # Original high-resolution frame
            # =================================================

            annotated_frame = (
                frame.copy()
            )


            # =================================================
            # Draw objects
            # =================================================

            for detection in detections:

                x1, y1, x2, y2 = (
                    detection["bbox"]
                )


                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)


                # ---------------------------------------------
                # Bounding box
                # ---------------------------------------------

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


                class_name = (
                    detection["class_name"]
                )

                confidence = (
                    detection["confidence"]
                )

                track_id = (
                    detection.get(
                        "track_id"
                    )
                )


                # ---------------------------------------------
                # Label
                # ---------------------------------------------

                if confidence is None:

                    if track_id is not None:

                        label = (
                            f"{class_name} "
                            f"ID:{track_id}"
                        )

                    else:

                        label = (
                            f"{class_name}"
                        )


                elif track_id is not None:

                    label = (
                        f"{class_name} "
                        f"ID:{track_id} "
                        f"{confidence:.2f}"
                    )

                else:

                    label = (
                        f"{class_name} "
                        f"{confidence:.2f}"
                    )


                label_position = (
                    x1,
                    max(
                        y1 - 10,
                        20
                    )
                )


                # Black outline
                cv2.putText(
                    annotated_frame,
                    label,
                    label_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    4
                )


                # White text
                cv2.putText(
                    annotated_frame,
                    label,
                    label_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )


                # ---------------------------------------------
                # Predicted center
                # ---------------------------------------------

                predicted_x = (
                    detection.get(
                        "predicted_x"
                    )
                )

                predicted_y = (
                    detection.get(
                        "predicted_y"
                    )
                )


                if (
                    predicted_x is not None
                    and
                    predicted_y is not None
                ):

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


            # =================================================
            # Timestamp
            # =================================================

            timestamp = (
                datetime.now().isoformat()
            )


            # =================================================
            # Detection CSV
            # =================================================

            if logger is not None:

                for detection in detections:

                    x1, y1, x2, y2 = (
                        detection["bbox"]
                    )


                    width = (
                        x2 - x1
                    )

                    height = (
                        y2 - y1
                    )


                    x_center = (
                        x1 + x2
                    ) / 2

                    y_center = (
                        y1 + y2
                    ) / 2


                    logger.log_detection(
                        {
                            "frame":
                                frame_number,

                            "timestamp":
                                timestamp,

                            "track_id":
                                detection.get(
                                    "track_id"
                                ),

                            "class_id":
                                detection[
                                    "class_id"
                                ],

                            "class_name":
                                detection[
                                    "class_name"
                                ],

                            "confidence":
                                detection[
                                    "confidence"
                                ],

                            "x_center":
                                x_center,

                            "y_center":
                                y_center,

                            "width":
                                width,

                            "height":
                                height,

                            "x1":
                                x1,

                            "y1":
                                y1,

                            "x2":
                                x2,

                            "y2":
                                y2,

                            "predicted_x":
                                detection.get(
                                    "predicted_x"
                                ),

                            "predicted_y":
                                detection.get(
                                    "predicted_y"
                                ),

                            "velocity_x":
                                detection.get(
                                    "velocity_x"
                                ),

                            "velocity_y":
                                detection.get(
                                    "velocity_y"
                                )
                        }
                    )


            # =================================================
            # Processing latency
            # =================================================

            processing_end = (
                time.perf_counter()
            )


            processing_latency = (
                processing_end
                - processing_start
            ) * 1000


            # =================================================
            # Monitoring
            # =================================================

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


            # =================================================
            # Monitoring CSV
            # =================================================

            if (
                updated
                and
                logger is not None
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


            # =================================================
            # Draw monitoring
            # =================================================

            annotated_frame = (
                display.draw_monitoring(
                    annotated_frame,
                    monitor
                )
            )


            # =================================================
            # Save video
            # =================================================

            if video_writer is not None:

                video_writer.write(
                    annotated_frame
                )


            # =================================================
            # Display
            # =================================================

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