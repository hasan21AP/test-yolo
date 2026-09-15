from ultralytics import YOLO


class ModelRunner:

    def __init__(
        self,
        model_path,
        device="cpu",
        imgsz=640,
        conf=0.35
    ):
        self.model_path = model_path
        self.device = device
        self.imgsz = imgsz
        self.conf = conf

        self.model = YOLO(self.model_path)

    def detect(self, frame):

        results = self.model.predict(
            source=frame,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False
        )

        return results