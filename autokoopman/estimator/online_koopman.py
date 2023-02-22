from typing import Optional

import autokoopman.core.estimator as kest
import autokoopman.core.system as asys
import autokoopman.core.system as ksys

import numpy as np

from odmd import OnlineDMD


class OnlineKoopmanEstimator(kest.GradientEstimator, kest.OnlineEstimator):
    def __init__(self, observables, weighting: float = 0.9) -> None:
        self._wf = weighting
        self.obs = observables
        self._odmd = None

    def fit_gradient(self, X: np.ndarray, Y: np.ndarray, U: Optional[np.ndarray] = None) -> None:
        # the size of this determines the p...
        assert U is None, f"ODMD doesn't work for systems with input (for now)"
        n = X.shape[0]
        G = np.array([self.obs(xi).flatten() for xi in X.T]).T
        Gp = np.array([self.obs(xi).flatten() for xi in Y.T]).T
        self._odmd = OnlineDMD(n, self._wf)
        self._odmd.initialize(G, Gp)

    def model(self) -> asys.System:
        """
        packs the learned linear transform into a continuous linear system
        """
        def grad_func(t, x, i):
            obs = (self.obs(x).flatten())[np.newaxis, :]
            return np.real(self._odmd.A @ obs.T).flatten()[: len(x)]
        return ksys.GradientContinuousSystem(grad_func, self.names)

    def update_single(self, x: np.ndarray, y: np.ndarray, u: Optional[np.ndarray] = None):
        assert u is None, f"ODMD doesn't work for systems with input (for now)"
        self._odmd.update(x, y)
