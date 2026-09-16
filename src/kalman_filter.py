import time

import cv2
import numpy as np


class KalmanFilter:

    def __init__(
        self,
        initial_box,
        process_noise=1e-3,
        measurement_noise=1e-1
    ):

        # State:
        # [cx, cy, w, h, vx, vy, vw, vh]

        self.kf = cv2.KalmanFilter(
            8,
            4
        )

        # Measurement:
        # [cx, cy, w, h]

        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0]
            ],
            dtype=np.float32
        )

        self.process_noise = process_noise
        self.base_measurement_noise = measurement_noise

        self.kf.processNoiseCov = (
            np.eye(
                8,
                dtype=np.float32
            )
            * self.process_noise
        )

        self.kf.errorCovPost = np.eye(
            8,
            dtype=np.float32
        )

        self.last_time = time.perf_counter()

        self.init_state(
            initial_box
        )


    # =====================================================
    # Initialize state
    # =====================================================

    def init_state(self, box):

        x1, y1, x2, y2 = box

        width = max(
            1.0,
            x2 - x1
        )

        height = max(
            1.0,
            y2 - y1
        )

        center_x = (
            x1 + x2
        ) / 2.0

        center_y = (
            y1 + y2
        ) / 2.0


        self.kf.statePost = np.array(
            [
                [center_x],
                [center_y],
                [width],
                [height],

                [0.0],
                [0.0],
                [0.0],
                [0.0]
            ],
            dtype=np.float32
        )


    # =====================================================
    # Update transition matrix
    # =====================================================

    def update_transition_matrix(self):

        current_time = time.perf_counter()

        dt = (
            current_time
            - self.last_time
        )

        self.last_time = current_time


        if dt <= 0:
            dt = 0.001


        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, dt, 0,  0,  0],
                [0, 1, 0, 0, 0,  dt, 0,  0],
                [0, 0, 1, 0, 0,  0,  dt, 0],
                [0, 0, 0, 1, 0,  0,  0,  dt],

                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]
            ],
            dtype=np.float32
        )


    # =====================================================
    # Predict
    # =====================================================

    def predict(self):

        self.update_transition_matrix()

        prediction = (
            self.kf.predict()
        )


        center_x = float(
            prediction[0, 0]
        )

        center_y = float(
            prediction[1, 0]
        )

        width = max(
            1.0,
            float(prediction[2, 0])
        )

        height = max(
            1.0,
            float(prediction[3, 0])
        )


        return self.make_box(
            center_x,
            center_y,
            width,
            height
        )


    # =====================================================
    # Correct using YOLO measurement
    # =====================================================

    def update(
        self,
        box,
        confidence
    ):

        x1, y1, x2, y2 = box

        width = max(
            1.0,
            x2 - x1
        )

        height = max(
            1.0,
            y2 - y1
        )

        center_x = (
            x1 + x2
        ) / 2.0

        center_y = (
            y1 + y2
        ) / 2.0


        # Higher YOLO confidence
        # = trust measurement more

        adaptive_noise = (
            self.base_measurement_noise
            / max(
                0.1,
                confidence
            )
        )


        self.kf.measurementNoiseCov = (
            np.eye(
                4,
                dtype=np.float32
            )
            * adaptive_noise
        )


        measurement = np.array(
            [
                [center_x],
                [center_y],
                [width],
                [height]
            ],
            dtype=np.float32
        )


        estimate = (
            self.kf.correct(
                measurement
            )
        )


        estimated_x = float(
            estimate[0, 0]
        )

        estimated_y = float(
            estimate[1, 0]
        )

        estimated_width = max(
            1.0,
            float(estimate[2, 0])
        )

        estimated_height = max(
            1.0,
            float(estimate[3, 0])
        )


        return self.make_box(
            estimated_x,
            estimated_y,
            estimated_width,
            estimated_height
        )


    # =====================================================
    # Get position
    # =====================================================

    def get_position(self):

        state = self.kf.statePost

        return (
            float(state[0, 0]),
            float(state[1, 0])
        )


    # =====================================================
    # Get velocity
    # =====================================================

    def get_velocity(self):

        state = self.kf.statePost

        return (
            float(state[4, 0]),
            float(state[5, 0])
        )


    # =====================================================
    # Make bounding box
    # =====================================================

    def make_box(
        self,
        center_x,
        center_y,
        width,
        height
    ):

        return [
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0
        ]