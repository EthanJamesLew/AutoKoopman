"""
Model has:

* a set of hyperparameters
"""
import abc

from typing import Optional

import numpy as np

import autokoopman.core.trajectory as atraj
import autokoopman.core.system as asys


class TrajectoryEstimator(abc.ABC):
    @abc.abstractmethod
    def fit(self, X: atraj.TrajectoriesData) -> None:
        pass

    @property
    @abc.abstractmethod
    def model(self) -> asys.System:
        pass


class NextStepEstimator(TrajectoryEstimator):
    """Estimator of discrete time dynamical systems

    Requires that the data be uniform time
    """

    @abc.abstractmethod
    def fit_next_step(self, X: np.ndarray, Y: np.ndarray) -> None:
        """an alternative fit method that uses a trajectories data structure"""
        pass

    def fit(self, X: atraj.TrajectoriesData) -> None:
        assert isinstance(X, atraj.UniformTimeTrajectoriesData), f"X must be uniform time"
        self.fit_next_step(*X.next_step_matrices)
        self.sampling_period = X.sampling_period
        self.names = X.state_names


