"""This file contains stubs for implementation of the Kalman filter"""
from model import KFModel
from numpy.typing import ArrayLike
import numpy as np

class KalmanFilter:

    def __init__(self, params: KFModel):
        self.params = params
        """Other initializations can go here"""

    def predict(self) -> ArrayLike:
        """
        This function, when implemented, should predict one step of the Kalman filter and return the predicted state vector
        """

    def correct(self, meas) -> ArrayLike:
        """
        This function, when implemented, should implement one correction step of the Kalman filter and return the corrected state vector 
        """
    
    def run_n_steps(self, N:int , measurements: ArrayLike) -> ArrayLike:
        """Given N (number of steps) and a 2xN measurement, runs N steps of the KF and returns a 4 x N trajectory prediction

        Args:
            N ([int]): Number of steps
            measurements ([ArrayLike]): Measurement array

        Returns:
            ArrayLike: Trajectory
        """
    