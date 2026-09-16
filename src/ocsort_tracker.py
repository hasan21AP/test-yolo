class OCSortTracker:

    def __init__(self, model_runner):
        self.model_runner = model_runner

    def update(self, frame):

        results = self.model_runner.model.track(
            source=frame,
            device=self.model_runner.device,
            imgsz=self.model_runner.imgsz,
            conf=self.model_runner.conf,

            tracker="ocsort.yaml",

            persist=True,
            verbose=False
        )

        return results