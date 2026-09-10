import time
import os

import psutil


class Monitor:

    def __init__(self, update_interval=5.0):
        self.update_interval = update_interval

        self.process = psutil.Process(os.getpid())

        self.last_update_time = time.perf_counter()
        self.frame_count = 0

        self.fps = 0.0
        self.inference_latency = 0.0
        self.model_call_latency = 0.0
        self.processing_latency = 0.0

        self.cpu_process = 0.0
        self.cpu_system = 0.0
        self.ram_process = 0.0
        self.ram_system = 0.0

        self.detections = 0

        self.process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

    def update(
        self,
        inference_latency,
        model_call_latency,
        processing_latency,
        detections
    ):
        self.frame_count += 1

        current_time = time.perf_counter()
        elapsed_time = current_time - self.last_update_time

        if elapsed_time < self.update_interval:
            return False

        self.fps = self.frame_count / elapsed_time

        self.inference_latency = inference_latency
        self.model_call_latency = model_call_latency
        self.processing_latency = processing_latency

        self.cpu_process = self.process.cpu_percent(interval=None)
        self.cpu_system = psutil.cpu_percent(interval=None)

        self.ram_process = self.process.memory_info().rss / (1024 ** 2)
        self.ram_system = psutil.virtual_memory().percent

        self.detections = detections

        self.frame_count = 0
        self.last_update_time = current_time

        return True