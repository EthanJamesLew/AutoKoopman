import abc

import numpy as np


class NextStepEstimator(abc.ABC):
    """Estimator of discrete time dynamical systems
    """
    @abc.abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass