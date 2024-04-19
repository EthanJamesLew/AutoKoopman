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
    try:
        Atilde = Yp @ V @ np.linalg.inv(Sigma) @ U.conj().T
    except:
        Atilde = Yp @ V @ np.linalg.pinv(Sigma) @ U.conj().T

    return Atilde[:, :state_size], Atilde[:, state_size:]


def wdmdc(X, Xp, U, r, W):
    """Weighted Dynamic Mode Decomposition with Control (wDMDC)"""
    # we allow weights to be either in diag or full representation
    W = np.diag(W) if len(np.array(W).shape) == 2 else W

    if U is not None:
        Y = np.hstack((X, U)).T
    else:
        Y = X.T
    Yp = Xp.T
    Y, Yp = W * Y, W * Yp
    state_size = Yp.shape[0]

    # compute Atilde
    U, Sigma, V = np.linalg.svd(Y, False)
    U, Sigma, V = U[:, :r], np.diag(Sigma)[:r, :r], V.conj().T[:, :r]

    # get the transformation
    # get the transformation
    try:
        Atilde = Yp @ V @ np.linalg.inv(Sigma) @ U.conj().T
    except:
        Atilde = Yp @ V @ np.linalg.pinv(Sigma) @ U.conj().T
    return Atilde[:, :state_size], Atilde[:, state_size:]


def swdmdc(X, Xp, U, r, Js, W):
    """State Weighted Dynamic Mode Decomposition with Control (wDMDC)"""
    import cvxpy as cp
    import warnings
    
    assert len(W.shape) == 2, "weights must be 2D for snapshot x state"

    if U is not None:
        Y = np.hstack((X, U))
    else:
        Y = X
    Yp = Xp
    state_size = Yp.shape[1]

    n_snap, n_obs = Yp.shape
    n_inps = U.shape[1] if U is not None else 0
    _, n_states = Js[0].shape

    # so the objective isn't numerically unstable
    sf = (1.0 / n_snap)

    # koopman operator
    K = cp.Variable((n_obs, n_obs + n_inps))

    # SW-eDMD objective
    weights_obj = np.vstack([(np.clip(np.abs(J), 0.0, 1.0) @ w) for J, w in zip(Js, W)]).T 
    P = sf * cp.multiply(weights_obj, Yp.T - K @ Y.T)
    # add regularization 
    objective = cp.Minimize(cp.sum_squares(P) + 1E-4 * 1.0 / (n_obs**2) * cp.norm(K, "fro"))

    # unconstrained problem
    constraints = None

    # SW-eDMD problem
    prob = cp.Problem(objective, constraints)

    # solve for the SW-eDMD Koopman operator
    try:
        _ = prob.solve(solver=cp.CLARABEL)
        #_ = prob.solve(solver=cp.ECOS)
        if K.value is None:
            raise Exception("SW-eDMD (cvxpy) Optimization failed to converge.")
    except:
        warnings.warn("SW-eDMD (cvxpy) Optimization failed to converge. Switching to unweighted DMDc.")
        return dmdc(X, Xp, U, r)
    
    # get the transformation
    Atilde = K.value 
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
    :param weights: observation weights (optional)

    References
        Proctor, J. L., Brunton, S. L., & Kutz, J. N. (2018). Generalizing Koopman theory to allow for inputs and control. SIAM Journal on Applied Dynamical Systems, 17(1), 909-930.

        See https://epubs.siam.org/doi/pdf/10.1137/16M1062296 for more details
    """

    def __init__(self, observables, sampling_period, dim, rank, weights=None, **kwargs):
        super().__init__(weights=weights, **kwargs)
        self.dim = dim
        self.obs = observables
        self.rank = int(rank)
        self._A, self._B = None, None

    def fit_next_step(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        U: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        """fits the discrete system model

        calls DMDC after building out the observables
        """
        G = np.array([self.obs(xi).flatten() for xi in X.T])
        Gp = np.array([self.obs(xi).flatten() for xi in Y.T])
        if weights is None:
            self._A, self._B = dmdc(G, Gp, U.T if U is not None else U, self.rank)
        else:
            # TODO: change this condition to be more accurate
            if False:  # len(weights[0].shape) == 1:
                self._A, self._B = wdmdc(
                    G, Gp, U.T if U is not None else U, self.rank, weights
                )
            else:
                self._A, self._B = swdmdc(
                    G,
                    Gp,
                    U.T if U is not None else U,
                    self.rank,
                    [self.obs.obs_grad(xi) for xi in X.T],
                    weights,
                )
        self._has_input = U is not None

    @property
    def model(self) -> ksys.System:
        """
        packs the learned linear transform into a discrete linear system
        """
        return ksys.KoopmanStepDiscreteSystem(
            self._A, self._B, self.obs, self.names, self.dim, self.scaler
        )


class KoopmanContinuousEstimator(kest.GradientEstimator):
    """Koopman Continuous Estimator

    :param observables: function that returns the observables of the system state
    :param dim: dimension of the system state
    :param rank: rank of the DMDc
    :param weights: observation weights (optional)

    References
        Proctor, J. L., Brunton, S. L., & Kutz, J. N. (2018). Generalizing Koopman theory to allow for inputs and control. SIAM Journal on Applied Dynamical Systems, 17(1), 909-930.

        See https://epubs.siam.org/doi/pdf/10.1137/16M1062296 for more details

    """

    def __init__(self, observables, dim, rank, weights=None, **kwargs):
        super().__init__(weights=weights, **kwargs)
        self.dim = dim
        self.obs = observables
        self.rank = int(rank)
        self._A, self._B = None, None

    def fit_gradient(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        U: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        """fits the gradient system model

        calls DMDC after building out the observables
        """
        G = np.array([self.obs(xi).flatten() for xi in X.T])
        Gp = np.array([self.obs(xi).flatten() for xi in Y.T])
        if weights is None:
            self._A, self._B = dmdc(G, Gp, U.T if U is not None else U, self.rank)
        else:
            self._A, self._B = wdmdc(
                G, Gp, U.T if U is not None else U, self.rank, weights
            )
        self._has_input = U is not None

    @property
    def model(self) -> ksys.System:
        """
        packs the learned linear transform into a continuous linear system
        """
        return ksys.KoopmanGradientContinuousSystem(
            self._A, self._B, self.obs, self.names, self.dim, self.scaler
        )
