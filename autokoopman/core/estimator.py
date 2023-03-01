"""
Model has:

* a set of hyperparameters
"""
import abc
from typing import Optional

import numpy as np

import autokoopman.core.system as asys
import autokoopman.core.trajectory as atraj


class TrajectoryEstimator(abc.ABC):
    @staticmethod
    def dynamics_from_trajs(X: atraj.UniformTimeTrajectoriesData):
        return X.differentiate
    
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
    @staticmethod
    def dynamics_from_trajs(X: atraj.UniformTimeTrajectoriesData):
        return X.next_step_matrices
    
    def __init__(self) -> None:
        self.names = None

    @abc.abstractmethod
    def fit_next_step(
        self, X: np.ndarray, Y: np.ndarray, U: Optional[np.ndarray] = None
    ) -> None:
        assert len(X) == len(Y) == len(U), "X, Y, and U must be the same length"
        """an alternative fit method that uses a trajectories data structure"""
        pass

    def fit(self, X: atraj.TrajectoriesData) -> None:
        assert isinstance(
            X, atraj.UniformTimeTrajectoriesData
        ), "X must be uniform time"
        self.fit_next_step(*self.dynamics_from_trajs(X))
        self.sampling_period = X.sampling_period
        self.names = X.state_names



class GradientEstimator(TrajectoryEstimator):
    """Estimator of discrete time dynamical systems

    Requires that the data be uniform time
    """
    @abc.abstractmethod
    def fit_gradient(
        self, X: np.ndarray, Y: np.ndarray, U: Optional[np.ndarray] = None
    ) -> None:
        """an alternative fit method that uses a trajectories data structure"""
        pass

    def fit(self, X: atraj.TrajectoriesData) -> None:
        assert isinstance(
            X, atraj.UniformTimeTrajectoriesData
        ), "X must be uniform time"
        self.fit_gradient(*self.dynamics_from_trajs(X))
        self.sampling_period = X.sampling_period
        self.names = X.state_names


class OnlineEstimator(TrajectoryEstimator):
    """abstract Estimator for streaming data, 
    
    online (streaming) estimators have a concept of initial fit and updates,
    which are ideally more efficient than calling fit every update. Further,
    cases are added for single observation (rank 1) update and batch updating.
    Batch updating is also available for implementing windowing estimators. 
    """
    @abc.abstractmethod
    def update_single(
        self, x: np.ndarray, y: np.ndarray, u: Optional[np.ndarray] = None
    ):
        """update given a single observation (x, y, and u are 1D arrays)"""
        pass

    def update_batch(
        self, X: np.ndarray, Y: np.ndarray, U: Optional[np.ndarray] = None
    ):
        """update given multiple observations (x, y, and u are 2D arrays)
        
        default behavior is to called update_single for each observation.
        """
        if U is None:
            U = [None]*X.shape[1]
        assert X.shape[1] == Y.shape[1] == len(U), "X and Y must be the same length"
        for x, y, u in zip(X.T, Y.T, U):
            self.update_single(x, y, u)

    def update(self,  X: atraj.TrajectoriesData):
        """main update methods for TrajectoriesData"""
        assert isinstance(
            X, atraj.UniformTimeTrajectoriesData
        ), "X must be uniform time"
        self.update_batch(*self.dynamics_from_trajs(X))

    def initialize(
        self, X: atraj.TrajectoriesData
    ) -> None:
        """online estimator initialization fit
        
        Default behavior is to call fit.
        """
        self.fit(X)
