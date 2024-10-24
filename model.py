"""
This file contains models for the Kalman Filter and UKF/Particle filters
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike

class FwdModel(ABC):
    """
    This is an **abstract** class defined for convenience. DO NOT try to create objects from this class.
    See below for KFModel and NonLinearKFModel, respectively.
    """
    @abstractmethod
    def __init__(self,
        delta: Optional[float] = None,
        Lambda: Optional[ArrayLike] = None,
        Q: Optional[ArrayLike] = None,
        R: Optional[ArrayLike] = None,
        ) -> None:

        # delta
        self.__delta = delta if delta is not None else 0.1

        # Forward model: starting point 
        self.__Lambda = Lambda if Lambda is not None else 1e-02*np.identity(2)
        
        # Process noise
        default_process_noise_std = 5e-03
        self.__Q = Q if Q is not None else (default_process_noise_std**2)*np.identity(4)

        # Measurement noise
        default_meas_noise_std = 1e-01
        self.__R = R if R is not None else (default_meas_noise_std**2)*np.identity(2)

    @property
    def delta(self) -> float:
        return self.__delta

    @property
    def Lambda(self) -> ArrayLike:
        return self.__Lambda

    @property
    def Q(self) -> ArrayLike:
        return self.__Q

    @property
    def R(self) -> ArrayLike:
        return self.__R

    @abstractmethod
    def field(self, x: ArrayLike, y: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Get the electric field at a bunch of locations specified in (x, y)"""
        pass

    def gen_data(self, max_N:int) -> Tuple[int, ArrayLike, ArrayLike]:
        """
        Generate a ground-truth trajectory and measurements from a  model
            - max_N is the largest N for which the trajectory will be computed. 
            - The returned values are N; the actual number of steps in the trajectory, where
                If either  the x or y-coordinate of the trajectory hits 1.0 
                then we terminate process and return N,
                else run max_N steps.
        """
        pass
    
    
class KFModel(FwdModel):
    
    """
    This class contains the models required for the Kalman filter of Part I
    
    For testing with the model specified, just instantiate the class with its default
    parameters. In the GradeScope tests, other parameters will be used.

    You use this model as follows:

        model_params = KFModel()
        model_params.Lambda*(...) + model_params.G*(...), etc.

        model.G returns the matrix G, and model.field(x, y) returns the actual field value at (x, y)

    """
    def __init__(self,
        delta: Optional[float] = None,
        G: Optional[ArrayLike] = None,
        Lambda: Optional[ArrayLike] = None,
        Q: Optional[ArrayLike] = None,
        R: Optional[ArrayLike] = None,
        ) -> None:
        self.__G = G or np.array([[1, 1], [1, -1]])
        super().__init__(delta=delta, Lambda=Lambda, Q=Q, R=R)
    
    @property
    def G(self) -> ArrayLike:
        return self.__G

    def field(self, x: ArrayLike, y: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """
        Get the electric field at a set of points x, y 
        (x and y must be the same shape)
        """
        u, v = self.G @ np.vstack((x.reshape(-1), y.reshape(-1)))
        return u.reshape(x.shape), v.reshape(y.shape)
        
    def gen_data(self, max_N:int) -> Tuple[int, ArrayLike, ArrayLike]:
        """
        Generate a ground-truth trajectory and measurements from a  model
            - max_N is the largest N for which the trajectory will be computed. 
            - The returned values are N; the actual number of steps in the trajectory, where
                If either  the x or y-coordinate of the trajectory hits 1.0 
                then we terminate process and return N,
                else run max_N steps.
        """
        trajectory = np.zeros((4, max_N), dtype=float)
        measurements = np.zeros((2, max_N), dtype=float)
        trajectory[0:2, 0] = np.random.multivariate_normal(
                                                mean = np.zeros(2), 
                                                cov = self.Lambda
                                            )
        while(abs(trajectory[0, 0]) >=  1 or abs(trajectory[1, 0]) >= 1):
            trajectory[0:2, 0] = np.random.multivariate_normal(
                                                mean = np.zeros(2), 
                                                cov = self.Lambda
                                            ) 
        
        measurements[:, 0] = trajectory[0:2, 0] + np.random.multivariate_normal(
                                                    mean = np.zeros(2), 
                                                    cov = self.R
                                                    ) 
        for n in range(1, max_N):
            a = np.array(self.field(
                                x=trajectory[0, n-1], 
                                y=trajectory[1, n-1]
                            )
                        )
            trajectory[2:4, n] = trajectory[2:4, n-1] + a*self.delta
            trajectory[0:2, n] = trajectory[0:2, n-1] + trajectory[2:4, n-1]*self.delta + (a*self.delta*self.delta)/2
            trajectory[:, n] = trajectory[:,n] + np.random.multivariate_normal(
                                                    mean = np.zeros(4), 
                                                    cov = self.Q
                                                ) 
            measurements[:, n] = trajectory[0:2, n] + np.random.multivariate_normal(
                                                    mean = np.zeros(2), 
                                                    cov = self.R
                                                ) 
            if (abs(trajectory[0, n]) > 1 or abs(trajectory[1, n]) > 1):
                return n, trajectory[:, :n], measurements[:, :n]

        return max_N, trajectory, measurements


