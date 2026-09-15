import numpy as np


class KalmanFilter:

    def __init__(
        self,
        x,
        y,
        dt=1.0,
        process_noise=1.0,
        measurement_noise=10.0
    ):

        # State:
        # [x, y, vx, vy]
        self.x = np.array(
            [
                [x],
                [y],
                [0.0],
                [0.0]
            ],
            dtype=float
        )

        # State transition matrix
        self.F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ],
            dtype=float
        )

        # Measurement matrix
        self.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ],
            dtype=float
        )

        # State uncertainty
        self.P = np.eye(4) * 100.0

        # Process noise
        self.Q = np.eye(4) * process_noise

        # Measurement noise
        self.R = np.eye(2) * measurement_noise

        # Identity matrix
        self.I = np.eye(4)

    def predict(self):

        # Predict state
        self.x = self.F @ self.x

        # Predict uncertainty
        self.P = (
            self.F
            @ self.P
            @ self.F.T
            + self.Q
        )

        return self.get_position()

    def update(self, measured_x, measured_y):

        # Measurement
        z = np.array(
            [
                [measured_x],
                [measured_y]
            ],
            dtype=float
        )

        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = (
            self.H
            @ self.P
            @ self.H.T
            + self.R
        )

        # Kalman Gain
        K = (
            self.P
            @ self.H.T
            @ np.linalg.inv(S)
        )

        # Update state
        self.x = self.x + K @ y

        # Update uncertainty
        self.P = (
            self.I - K @ self.H
        ) @ self.P

        return self.get_position()

    def get_position(self):

        x = float(self.x[0, 0])
        y = float(self.x[1, 0])

        return x, y

    def get_velocity(self):

        vx = float(self.x[2, 0])
        vy = float(self.x[3, 0])

        return vx, vy