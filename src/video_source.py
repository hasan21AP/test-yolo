import cv2


class VideoSource:

    def __init__(self, source):
        self.source = source
        self.cap = None

        self.width = 0
        self.height = 0
        self.fps = 0.0

    def open(self):

        # Local webcam
        if isinstance(self.source, int):

            self.cap = cv2.VideoCapture(
                self.source,
                cv2.CAP_DSHOW
            )

            self.cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*"MJPG")
            )

            # Request high resolution
            self.cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                1920
            )

            self.cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                1080
            )

        # Video file / IP camera
        else:

            self.cap = cv2.VideoCapture(
                self.source
            )

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open video source: {self.source}"
            )

        self.width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        self.height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        self.fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        print(
            f"Source resolution: "
            f"{self.width}x{self.height}"
        )

    def read(self):
        return self.cap.read()

    def release(self):

        if self.cap is not None:
            self.cap.release()
            self.cap = None