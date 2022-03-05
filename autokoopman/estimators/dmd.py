import numpy as np
from pydmd import DMD  # type: ignore
from pydmd.dmd import compute_tlsq  # type: ignore

import autokoopman.core.estimator as aest
import autokoopman.core.system as asys


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


class DMDEstimator(aest.NextStepEstimator):
    def fit_next_step(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.dmd = MultiDMD(svd_rank=-1)
        self.dmd.fit_multi(X, Y)

    @property
    def model(self) -> asys.System:
        def step_func(t, x):
            return np.real(self.dmd.predict(np.atleast_2d(x).T)).T

        return asys.StepDiscreteSystem(step_func, self.names)
