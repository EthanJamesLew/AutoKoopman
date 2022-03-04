import abc

import numpy as np

import autokoopman.trajectory as atraj

class NextStepEstimator(abc.ABC):
    """Estimator of discrete time dynamical systems

    Requires that the data be uniform time
    """
    @abc.abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass

    def fit_trajs(self, X: atraj.UniformTimeTrajectoriesData) -> None:
        """an alternative fit method that uses a trajectories data structure"""
        pass

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class DMDEstimator(NextStepEstimator):
    ...