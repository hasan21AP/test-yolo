class SingleIoUTracker:

    def __init__(
        self,
        iou_threshold=0.25,
        max_lost_frames=10
    ):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames

        self.locked = False
        self.locked_box = None
        self.locked_class_id = None
        self.locked_class_name = None
        self.locked_confidence = 0.0

        self.lost_frames = 0

        self.track_id = None
        self.next_id = 1


    # =====================================================
    # IoU
    # =====================================================

    def calculate_iou(self, box_a, box_b):

        x1 = max(
            box_a[0],
            box_b[0]
        )

        y1 = max(
            box_a[1],
            box_b[1]
        )

        x2 = min(
            box_a[2],
            box_b[2]
        )

        y2 = min(
            box_a[3],
            box_b[3]
        )

        intersection = (
            max(0, x2 - x1)
            *
            max(0, y2 - y1)
        )

        area_a = max(
            1.0,
            (box_a[2] - box_a[0])
            *
            (box_a[3] - box_a[1])
        )

        area_b = max(
            1.0,
            (box_b[2] - box_b[0])
            *
            (box_b[3] - box_b[1])
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union


    # =====================================================
    # Create output
    # =====================================================

    def make_output(self):

        if not self.locked:
            return []

        x1, y1, x2, y2 = (
            self.locked_box
        )

        center_x = (
            x1 + x2
        ) / 2

        center_y = (
            y1 + y2
        ) / 2

        return [
            {
                "bbox":
                    self.locked_box,

                "track_id":
                    self.track_id,

                "class_id":
                    self.locked_class_id,

                "class_name":
                    self.locked_class_name,

                "confidence":
                    self.locked_confidence,

                "predicted_x":
                    center_x,

                "predicted_y":
                    center_y,

                "velocity_x":
                    None,

                "velocity_y":
                    None
            }
        ]


    # =====================================================
    # Update
    # =====================================================

    def update(self, detections):

        # -------------------------------------------------
        # No active target
        # Select highest confidence detection
        # -------------------------------------------------

        if not self.locked:

            if len(detections) == 0:
                return []

            best_detection = max(
                detections,
                key=lambda detection:
                    detection["confidence"]
            )

            self.locked_box = (
                best_detection["bbox"]
            )

            self.locked_class_id = (
                best_detection["class_id"]
            )

            self.locked_class_name = (
                best_detection["class_name"]
            )

            self.locked_confidence = (
                best_detection["confidence"]
            )

            self.track_id = self.next_id
            self.next_id += 1

            self.lost_frames = 0
            self.locked = True

            return self.make_output()


        # -------------------------------------------------
        # Target already locked
        # Find highest IoU
        # -------------------------------------------------

        best_match = None
        best_iou = 0.0


        for detection in detections:

            # Keep same class
            if (
                detection["class_id"]
                != self.locked_class_id
            ):
                continue

            iou = self.calculate_iou(
                self.locked_box,
                detection["bbox"]
            )

            if iou > best_iou:

                best_iou = iou
                best_match = detection


        # -------------------------------------------------
        # Match found
        # -------------------------------------------------

        if (
            best_match is not None
            and
            best_iou >= self.iou_threshold
        ):

            self.locked_box = (
                best_match["bbox"]
            )

            self.locked_confidence = (
                best_match["confidence"]
            )

            self.lost_frames = 0


        # -------------------------------------------------
        # Detection lost
        # -------------------------------------------------

        else:

            self.lost_frames += 1

            if (
                self.lost_frames
                >= self.max_lost_frames
            ):

                self.reset()

                return []


        return self.make_output()


    # =====================================================
    # YOLO not running
    # =====================================================

    def predict_only(self):

        # No Kalman here.
        # Keep last known box.

        return self.make_output()


    # =====================================================
    # Reset
    # =====================================================

    def reset(self):

        self.locked = False
        self.locked_box = None

        self.locked_class_id = None
        self.locked_class_name = None

        self.locked_confidence = 0.0

        self.lost_frames = 0
        self.track_id = None