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
import autokoopman.system as asys


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


class SindyEstimator(TrajectoryEstimator):
    def __init__(self, sindy: Optional[ps.SINDy] = None):
        self._usr_sindy = sindy

    def fit(self, X: atraj.TrajectoriesData) -> None:
        if self._usr_sindy is None:
            self._model = ps.SINDy(feature_names=X.state_names,
                                                differentiation_method=ps.FiniteDifference(),
                                                optimizer=ps.SR3(threshold=0.04, thresholder="l1"),
                                                feature_library=ps.PolynomialLibrary(degree=3))

        else:
            self._model = self._usr_sindy
        self._model.fit([xi.states for xi in X], t=[xi.times for xi in X], multiple_trajectories=True)

    def predict(self, iv: np.ndarray, times: np.ndarray) -> atraj.Trajectory:
        ret = self._model.simulate(iv, times)
        return atraj.Trajectory(times, ret, self._model.feature_names)

    @property
    def model(self) -> asys.ContinuousSystem:
        gradient_f = lambda t, x: self._model.predict(np.atleast_2d(x))
        return asys.GradientContinuousSystem(gradient_f, self._model.feature_names)


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

    @property
    def model(self) -> asys.System:
        step_func = lambda t, x: np.real(self.dmd.predict(np.atleast_2d(x).T)).T
        return asys.StepDiscreteSystem(step_func, self.names)
