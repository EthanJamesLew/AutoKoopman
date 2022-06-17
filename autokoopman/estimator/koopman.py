import autokoopman.core.estimator as kest
import autokoopman.core.system as ksys
import numpy as np
from typing import Optional


def dmdc(X, Xp, U, r):
    if U is not None:
        Y = np.hstack((X, U)).T
    else:
        Y = X.T
    Yp = Xp.T
    state_size = Yp.shape[0]

    # compute Atilde
    U, Sigma, V = np.linalg.svd(Y, False)
    U, Sigma, V = U[:, :r], np.diag(Sigma)[:r, :r], V.conj().T[:, :r]

    # get the transformation
    Atilde = Yp @ V @ np.linalg.inv(Sigma) @ U.conj().T
    return Atilde[:, :state_size], Atilde[:, state_size:]


class KoopmanDiscEstimator(kest.NextStepEstimator):
    def __init__(self, observables, sampling_period, dim, rank):
        self.dim = dim
        self.obs = observables
        self.rank = rank

    def fit_next_step(
        self, X: np.ndarray, Y: np.ndarray, U: Optional[np.ndarray] = None
    ) -> None:
        G = np.array([self.obs(xi).flatten() for xi in X.T])
        Gp = np.array([self.obs(xi).flatten() for xi in Y.T])
        self._A, self._B = dmdc(G, Gp, U.T if U is not None else U, self.rank)
        self._has_input = U is not None

    @property
    def model(self) -> ksys.System:
        def step_func(t, x, i):
            obs = (self.obs(x).flatten())[np.newaxis, :]
            if self._has_input:
                return np.real(
                    self._A @ obs.T + self._B @ (i)[:, np.newaxis]
                ).flatten()[: self.dim]
            else:
                return np.real(self._A @ obs.T).flatten()[: len(x)]

        return ksys.StepDiscreteSystem(step_func, self.names)
