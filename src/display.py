import cv2


class Display:

    def __init__(self, window_name="Vehicle YOLO"):
        self.window_name = window_name

    def draw_monitoring(self, frame, monitor):
        lines = [
            f"FPS: {monitor.fps:.2f}",
            f"Inference: {monitor.inference_latency:.2f} ms",
            f"Model Call: {monitor.model_call_latency:.2f} ms",
            f"Processing: {monitor.processing_latency:.2f} ms",
            f"CPU Process: {monitor.cpu_process:.2f}%",
            f"CPU System: {monitor.cpu_system:.2f}%",
            f"RAM Process: {monitor.ram_process:.2f} MB",
            f"RAM System: {monitor.ram_system:.2f}%",
            f"Detections: {monitor.detections}"
        ]

        x = 20
        y = 30

        for line in lines:
            cv2.putText(
                frame,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            y += 25

        return frame

    def show(self, frame):
        cv2.imshow(self.window_name, frame)

    def should_quit(self):
        key = cv2.waitKey(1) & 0xFF
        return key == ord("q")

    def close(self):
        cv2.destroyAllWindows()