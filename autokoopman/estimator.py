"""
Model has:

* a set of hyperparameters
"""
import abc

from typing import Optional, Tuple

import numpy as np
import pysindy as ps

from pydmd import DMD
from pydmd.dmd import compute_tlsq

import autokoopman.trajectory as atraj


class TrajectoryEstimator(abc.ABC):
    @abc.abstractmethod
    def fit(self, X: atraj.TrajectoriesData) -> None:
        pass

    @abc.abstractmethod
    def predict(self, iv: np.ndarray, times: np.ndarray) -> atraj.Trajectory:
        pass


class NextStepEstimator(TrajectoryEstimator):
    """Estimator of discrete time dynamical systems

    Requires that the data be uniform time
    """

    @abc.abstractmethod
    def fit_next_step(self, X: np.ndarray, Y: np.ndarray) -> None:
        """an alternative fit method that uses a trajectories data structure"""
        pass

    @abc.abstractmethod
    def predict_next_step(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit(self, X: atraj.TrajectoriesData) -> None:
        assert isinstance(X, atraj.UniformTimeTrajectoriesData), f"X must be uniform time"
        self.fit_next_step(*X.next_step_matrices)
        self.sampling_period = X.sampling_period
        self.names = X.state_names

    def predict_traj(self, iv: np.ndarray, tspan: Tuple[float, float]) -> atraj.Trajectory:
        times = np.arange(tspan[0], tspan[1] + self.sampling_period, self.sampling_period)
        states = np.zeros((len(times), len(self.names)))
        states[0] = iv
        for idx, _ in enumerate(times[1:]):
            states[idx + 1] = self.predict_next_step(states[idx]).flatten()
        return atraj.Trajectory(times, states, self.names)

    def predict(self, iv: np.ndarray, times: np.ndarray) -> atraj.Trajectory:
        return self.predict_traj(iv, (float(np.min(times)), float(np.max(times))))


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


class MultiDMD(DMD):
    def fit_multi(self, X, Y):
        """
        Compute the Dynamic Modes Decomposition to the input data.
        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        self._snapshots, self._snapshots_shape = self._col_major_2darray(X)

        # n_samples = self._snapshots.shape[1]
        # X = self._snapshots[:, :-1]
        # Y = self._snapshots[:, 1:]

        X, Y = compute_tlsq(X, Y, self.tlsq_rank)
        self._svd_modes, _, _ = self.operator.compute_operator(X, Y)

        # Default timesteps
        # self._set_initial_time_dictionary(
        #    {"t0": 0, "tend": n_samples - 1, "dt": 1}
        # )

        self._b = self._compute_amplitudes()

        return self


class DMDEstimator(NextStepEstimator):
    def fit_next_step(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.dmd = MultiDMD(svd_rank=-1)
        self.dmd.fit_multi(X, Y)

    def predict_next_step(self, X: np.ndarray) -> np.ndarray:
        return np.real(self.dmd.predict(np.atleast_2d(X).T)).T
