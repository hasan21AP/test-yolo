import csv
from pathlib import Path


class CSVLogger:

    def __init__(
        self,
        monitoring_path="logs/monitoring.csv",
        detections_path="logs/detections.csv"
    ):

        self.monitoring_path = Path(monitoring_path)
        self.detections_path = Path(detections_path)

        # Create logs folder if it does not exist
        self.monitoring_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.detections_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Open new CSV files for every run
        self.monitoring_file = open(
            self.monitoring_path,
            "w",
            newline="",
            encoding="utf-8"
        )

        self.detections_file = open(
            self.detections_path,
            "w",
            newline="",
            encoding="utf-8"
        )

        # =================================================
        # Monitoring fields
        # =================================================

        self.monitoring_fields = [
            "timestamp",
            "fps",
            "inference_latency",
            "model_call_latency",
            "processing_latency",
            "cpu_process",
            "cpu_system",
            "ram_process",
            "ram_system",
            "detections"
        ]

        # =================================================
        # Detection + Tracking + Kalman fields
        # =================================================

        self.detection_fields = [
            "frame",
            "timestamp",

            "track_id",

            "class_id",
            "class_name",
            "confidence",

            "x_center",
            "y_center",
            "width",
            "height",

            "x1",
            "y1",
            "x2",
            "y2",

            "predicted_x",
            "predicted_y",

            "velocity_x",
            "velocity_y"
        ]

        # =================================================
        # CSV writers
        # =================================================

        self.monitoring_writer = csv.DictWriter(
            self.monitoring_file,
            fieldnames=self.monitoring_fields
        )

        self.detections_writer = csv.DictWriter(
            self.detections_file,
            fieldnames=self.detection_fields
        )

        # Write headers
        self.monitoring_writer.writeheader()
        self.detections_writer.writeheader()

    # =====================================================
    # Monitoring log
    # =====================================================

    def log_monitoring(self, data):

        self.monitoring_writer.writerow(data)
        self.monitoring_file.flush()

    # =====================================================
    # Detection log
    # =====================================================

    def log_detection(self, data):

        self.detections_writer.writerow(data)
        self.detections_file.flush()

    # =====================================================
    # Close files
    # =====================================================

    def close(self):

        self.monitoring_file.close()
        self.detections_file.close()