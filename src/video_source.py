import cv2


class VideoSource:

    def __init__(self, source):
        self.source = source
        self.cap = None

        self.width = 0
        self.height = 0
        self.fps = 0.0

    def open(self):

        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def read(self):
        ret, frame = self.cap.read()
        return ret, frame

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def release(self):

        if self.cap is not None:
            self.cap.release()
            self.cap = None