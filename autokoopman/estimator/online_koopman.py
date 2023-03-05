from typing import Optional

import autokoopman.core.estimator as kest
import autokoopman.core.system as asys
import autokoopman.core.system as ksys

import numpy as np


class OnlineKoopmanEstimator(kest.GradientEstimator, kest.OnlineEstimator):
    """online EDMD with optional inputs"""
    def __init__(self, observables, weighting: float = 0.9, **kwargs) -> None:
        super().__init__(**kwargs)
        self._wf = weighting
        self.obs = observables
        self._odmd = None
        self.has_input = False

    def fit_gradient(self, X: np.ndarray, Y: np.ndarray, U: Optional[np.ndarray] = None) -> None:
        # the size of this determines the p...
        #assert U is None, f"ODMD doesn't work for systems with input (for now)"
        from autokoopman.estimator.online.online import OnlineInputDMD
        
        G = np.array([self.obs(xi).flatten() for xi in X.T]).T
        Gp = np.array([self.obs(xi).flatten() for xi in Y.T]).T
        n = G.shape[0]
        if U is not None and not np.any(np.array(U.shape) == 0):
            m = U.shape[0]
            self.has_input = True
        else:
            m = None
        self._odmd = OnlineInputDMD(n, m, self._wf)
        self._odmd.initialize(G, Gp, U)
        self.dim = self._odmd.n

    @property
    def model(self) -> asys.System:
        """
        packs the learned linear transform into a continuous linear system
        """
        # KIC -- extract A, B matrices from G
        _A = self._odmd.A
        n, m = self._odmd.n, self._odmd.m  
        A, B = _A[:, :n], _A[:, n:(n+m)]
        return ksys.KoopmanGradientContinuousSystem(
            A, B, self.obs, self.names, self.dim, self.scaler
        )

    def update_single(self, x: np.ndarray, y: np.ndarray, u: Optional[np.ndarray] = None):
        if self.scaler is not None:
            _x, _y = self.scaler.transform(np.atleast_2d(x)).flatten(), self.scaler.transform(np.atleast_2d(y)).flatten()
        else:
            _x, _y = x, y
        self._odmd.update(self.obs(_x).flatten(), self.obs(_y).flatten(), u)
