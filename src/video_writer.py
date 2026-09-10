import cv2
from pathlib import Path


class VideoWriter:

    def __init__(
        self,
        output_path="outputs/output.mp4",
        fps=30.0,
        width=640,
        height=480,
        codec="mp4v"
    ):
        self.output_path = Path(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec

        self.writer = None

    def open(self):

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        fourcc = cv2.VideoWriter_fourcc(*self.codec)

        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.width, self.height)
        )

        if not self.writer.isOpened():
            raise RuntimeError(
                f"Failed to open video writer: {self.output_path}"
            )

    def write(self, frame):
        self.writer.write(frame)

    def is_opened(self):
        return self.writer is not None and self.writer.isOpened()

    def release(self):

        if self.writer is not None:
            self.writer.release()
            self.writer = None