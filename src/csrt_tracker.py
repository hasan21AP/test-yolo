import cv2


class CSRTTracker:

    def __init__(self):
        self.tracker = None
        self.initialized = False

    def _create_tracker(self):
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()

        if (
            hasattr(cv2, "legacy")
            and hasattr(cv2.legacy, "TrackerCSRT_create")
        ):
            return cv2.legacy.TrackerCSRT_create()

        raise RuntimeError(
            "CSRT tracker is not available in this OpenCV build."
        )

    def init(self, frame, bbox):
        """
        bbox format:
        [x1, y1, x2, y2]
        """

        x1, y1, x2, y2 = bbox

        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid CSRT bbox: {bbox}"
            )

        csrt_bbox = (
            x1,
            y1,
            width,
            height
        )

        self.tracker = self._create_tracker()

        self.tracker.init(
            frame,
            csrt_bbox
        )

        self.initialized = True

    def update(self, frame):
        if not self.initialized:
            return False, None

        success, bbox = self.tracker.update(frame)

        if not success:
            return False, None

        x, y, width, height = bbox

        x1 = x
        y1 = y
        x2 = x + width
        y2 = y + height

        return True, [
            x1,
            y1,
            x2,
            y2
        ]

    def reset(self):
        self.tracker = None
        self.initialized = False    