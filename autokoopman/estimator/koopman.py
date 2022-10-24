import autokoopman.core.estimator as kest
import autokoopman.core.system as ksys
import numpy as np
from typing import Optional


def dmdc(X, Xp, U, r):
    """Dynamic Mode Decomposition with Control (DMDC)

    DMD but extended to include control.

    References
        Proctor, J. L., Brunton, S. L., & Kutz, J. N. (2016). Dynamic mode decomposition with control. SIAM Journal on Applied Dynamical Systems, 15(1), 142-161.

        See https://arxiv.org/pdf/1409.6358.pdf for more details
    """
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
    """Koopman Discrete Estimator

    This methods implements a regularized form of Koopman with Inputs (KIC). It assumes that the input
    if piecewise constant, zeroing out some of the state + input transform.

    TODO: add other ways to implement KIC
    TODO: sampling period isn't used

    :param observables: function that returns the observables of the system state
    :param sampling_period: sampling period of the uniform time, discrete system
    :param dim: dimension of the system state
    :param rank: rank of the DMDc

    References
        Proctor, J. L., Brunton, S. L., & Kutz, J. N. (2018). Generalizing Koopman theory to allow for inputs and control. SIAM Journal on Applied Dynamical Systems, 17(1), 909-930.

        See https://epubs.siam.org/doi/pdf/10.1137/16M1062296 for more details
    """

    def __init__(self, observables, sampling_period, dim, rank):
        super().__init__()
        self.dim = dim
        self.obs = observables
        self.rank = int(rank)

    def fit_next_step(
        self, X: np.ndarray, Y: np.ndarray, U: Optional[np.ndarray] = None
    ) -> None:
        """fits the discrete system model

        calls DMDC after building out the observables
        """
        G = np.array([self.obs(xi).flatten() for xi in X.T])
        Gp = np.array([self.obs(xi).flatten() for xi in Y.T])
        self._A, self._B = dmdc(G, Gp, U.T if U is not None else U, self.rank)
        self._has_input = U is not None

    @property
    def model(self) -> ksys.System:
        """
        packs the learned linear transform into a discrete linear system
        """

        def step_func(t, x, i):
            obs = (self.obs(x).flatten())[np.newaxis, :]
            if self._has_input:
                return np.real(
                    self._A @ obs.T + self._B @ (i)[:, np.newaxis]
                ).flatten()[: self.dim]
            else:
                return np.real(self._A @ obs.T).flatten()[: len(x)]

        return ksys.StepDiscreteSystem(step_func, self.names)


class KoopmanContinuousEstimator(kest.GradientEstimator):
    """Koopman Continuous Estimator

    :param observables: function that returns the observables of the system state
    :param dim: dimension of the system state
    :param rank: rank of the DMDc

    References
        Proctor, J. L., Brunton, S. L., & Kutz, J. N. (2018). Generalizing Koopman theory to allow for inputs and control. SIAM Journal on Applied Dynamical Systems, 17(1), 909-930.

        See https://epubs.siam.org/doi/pdf/10.1137/16M1062296 for more details

    """

    def __init__(self, observables, dim, rank):
        super().__init__()
        self.dim = dim
        self.obs = observables
        self.rank = int(rank)

    def fit_gradient(
        self, X: np.ndarray, Y: np.ndarray, U: Optional[np.ndarray] = None
    ) -> None:
        """fits the gradient system model

        calls DMDC after building out the observables
        """
        G = np.array([self.obs(xi).flatten() for xi in X.T])
        Gp = np.array([self.obs(xi).flatten() for xi in Y.T])
        self._A, self._B = dmdc(G, Gp, U.T if U is not None else U, self.rank)
        self._has_input = U is not None

    @property
    def model(self) -> ksys.System:
        """
        packs the learned linear transform into a continuous linear system
        """

        def grad_func(t, x, i):
            obs = (self.obs(x).flatten())[np.newaxis, :]
            if self._has_input:
                return np.real(
                    self._A @ obs.T + self._B @ (i)[:, np.newaxis]
                ).flatten()[: self.dim]
            else:
                return np.real(self._A @ obs.T).flatten()[: len(x)]

        return ksys.GradientContinuousSystem(grad_func, self.names)
