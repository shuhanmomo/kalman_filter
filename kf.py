"""This file contains stubs for implementation of the Kalman filter"""

from model import KFModel
from numpy.typing import ArrayLike
import numpy as np


class KalmanFilter:

    def __init__(self, params: KFModel):
        self.params = params
        """Other initializations can go here"""
        self.An, self.Bn = self.initialize_matrixes()
        self.Q = self.params.Q
        self.R = self.params.R
        self.X = np.zeros(4)
        self.X[0:2] = np.random.multivariate_normal(
            mean=np.zeros(2), cov=self.params.Lambda
        )

        initial_position_cov = self.params.Lambda
        initial_velocity_cov = np.eye(2) * 1e-1
        self.P = np.block(
            [
                [
                    initial_position_cov,
                    np.zeros((2, 2)),
                ],
                [
                    np.zeros((2, 2)),
                    initial_velocity_cov,
                ],
            ]
        )

    def initialize_matrixes(self) -> ArrayLike:
        G = self.params.G
        delta = self.params.delta
        # Extract G11, G12, G21, G22 from the matrix G
        G11, G12 = G[0, 0], G[0, 1]
        G21, G22 = G[1, 0], G[1, 1]

        # Define the state transition matrix An (4x4)
        An = np.array(
            [
                [1 + delta**2 / 2 * G11, delta**2 / 2 * G12, delta, 0],
                [delta**2 / 2 * G21, 1 + delta**2 / 2 * G22, 0, delta],
                [delta * G11, delta * G12, 1, 0],
                [delta * G21, delta * G22, 0, 1],
            ]
        )

        # Define the observation matrix Bn (2x4)
        Bn = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        return An, Bn

    def predict(self) -> ArrayLike:
        """
        This function, when implemented, should predict one step of the Kalman filter and return the predicted state vector
        """
        self.X = self.An @ self.X  # predict state
        self.P = self.An @ self.P @ self.An.T + self.Q  # Predicted covariance
        return self.X

    def correct(self, meas) -> ArrayLike:
        """
        This function, when implemented, should implement one correction step of the Kalman filter and return the corrected state vector
        """
        y = (
            meas - self.Bn @ self.X
        )  # residual between measurement yn and predicted state
        S = self.Bn @ self.P @ self.Bn.T + self.R  # innovation covariance
        K = self.P @ self.Bn.T @ np.linalg.inv(S)  # kalman gain
        self.X = self.X + K @ y  # update state estimate

        I = np.eye(self.P.shape[0])  # Identity matrix
        self.P = (I - K @ self.Bn) @ self.P
        return self.X

    def run_n_steps(self, N: int, measurements: ArrayLike) -> ArrayLike:
        """Given N (number of steps) and a 2xN measurement, runs N steps of the KF and returns a 4 x N trajectory prediction

        Args:
            N ([int]): Number of steps
            measurements ([ArrayLike]): Measurement array

        Returns:
            ArrayLike: Trajectory
        """
        trajectory = np.zeros((4, N))
        for n in range(N):
            # Predict step
            predicted_state = self.predict()

            # Correct step using the measurement
            corrected_state = self.correct(measurements[:, n])

            # Store the corrected state in the trajectory
            trajectory[:, n] = corrected_state

        return trajectory
