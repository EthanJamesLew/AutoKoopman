import abc

from typing import Optional

import numpy as np
import pysindy as ps

import autokoopman.trajectory as atraj


class TrajectoryEstimator(abc.ABC):
    @abc.abstractmethod
    def fit(self, X: atraj.TrajectoriesData) -> None:
        pass

    @abc.abstractmethod
    def predict(self, iv: np.ndarray, times: np.ndarray) -> atraj.Trajectory:
        pass


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


class SindyEstimator(TrajectoryEstimator):
    def __init__(self, sindy: Optional[ps.SINDy] = None):
        self._usr_sindy = sindy

    def fit(self, X: atraj.TrajectoriesData) -> None:
        if self._usr_sindy is None:
            self._model = self.model = ps.SINDy(feature_names=X.state_names,
                                 differentiation_method=ps.FiniteDifference(),
                                 optimizer=ps.SR3(threshold=0.04, thresholder="l1"),
                                 feature_library=ps.PolynomialLibrary(degree=3))

        else:
            self._model = self._usr_sindy
        self._model.fit([xi.states for xi in X], t=[xi.times for xi in X], multiple_trajectories=True)

    def predict(self, iv: np.ndarray, times: np.ndarray) -> atraj.Trajectory:
        ret = self._model.simulate(iv, times)
        return atraj.Trajectory(times, ret, self._model.feature_names)


class DMDEstimator(NextStepEstimator):
    ...