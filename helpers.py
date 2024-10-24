#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from model import FwdModel


def read_measurements(file) -> ArrayLike:    
    """
    Reads a ground-truth trajectory or measurement (txt) file 
    Returns:
        ArrayLike: representing 
                        (xk; yk; vxk; vyk) 
                   or 
                        (\hat{x}k; \hat{y}k)
                   depending on the type of file passed.
    """ 
    return np.loadtxt(file)

def plot_field(model: FwdModel) -> plt.Axes:
    """
    Take a field model, plots the field and returns the axis

    Args:
        model (FwdModel): A Kalman filter model that returns the field

    Returns:
        plt.Axes: A matplotlib axes object that can be passed to `plot_measurement`
    """
    
    A,B = np.meshgrid(np.linspace(-1,1,20),np.linspace(-1,1,20))
    u,v = model.field(A,B)
    _, ax = plt.subplots()
    ax.quiver(A, B, u, v, alpha = 0.4)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_aspect('equal')
    return ax

def plot_measurement(
        ax: plt.Axes, 
        groundtruth_traj: ArrayLike, 
        predict_traj: Optional[ArrayLike] = None
        ):
    """Take a ground truth trajectory and a predicted trajectory and plots them on top of a field

    Args:
        ax (plt.Axes): An axis to plot on. Get this from `plot_field`, and this function will 
                       overlay the trajectory on this axis.
        groundtruth_traj (ArrayLike): Plot the ground truth trajectory
        predict_traj (ArrayLike): If specified, also plot a predicted trajectory.
    """
    ax.plot(groundtruth_traj[0], groundtruth_traj[1], '-rx', label='ground truth trajectory')
    if predict_traj is not None:
        ax.plot(predict_traj[0], predict_traj[1],'-bo', label='predicted trajectory')
    
    

    