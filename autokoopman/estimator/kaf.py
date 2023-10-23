import numpy as np
from scipy.sparse.linalg import eigs

import autokoopman.core.estimator as aest
import autokoopman.core.system as asys


class KAF:
    r"""
    Kernel Analog Filtering (KAF)
        
        The original analog forecasting method is given initial data, locate its closest analog among the historical points and 
        reports the historical value of the corresponding observable, shifted by the desired lead time. This has been achieved 
        through the use of kernel-based methods which result in data-driven prediction based on weighting all the historical 
        data according to their similarity to the initial data; this leads to algorithms which enforce continuity of the 
        forecast with respect to initial data.

    :param kernel: kernel function (currently uses GPy kernel)
    :param approx_rank: rank parameter
    :param eig_mult: value to add to eigenvalue, scaled from max value

    References
        Burov, D., Giannakis, D., Manohar, K., & Stuart, A. (2021). Kernel analog forecasting: 
        Multiscale test problems. Multiscale Modeling & Simulation, 19(2), 1011-1040.
    """
    def __init__(self, kernel=None, approx_rank=None, eig_mult=1E-4) -> None:
        if kernel is None:
            self.kernel = RBF(X.shape[1])
        else:
            self.kernel = kernel
        self.approx_rank = int(approx_rank)
        self.eig_mult = eig_mult

    def fit(self, X, Xp):
        self.X = X.T
        self.KXX = self.kernel.K(X, X)
        _lambda, self.subspace = eigs(self.KXX, k=self.approx_rank if self.approx_rank is not None else len(self.KXX)-1)
        self.eigvals = _lambda + max(_lambda) * self.eig_mult
        self.weights = ((Xp.T @ self.subspace) @ np.linalg.pinv(np.diag(self.eigvals))) @ self.subspace.conj().T
        
    def predict(self, Xtest):
        Xtest = Xtest.T
        KXY = self.kernel.K(self.X.T, Xtest.T)
        return (self.weights @ KXY).T


class KAFEstimator(aest.NextStepEstimator):
    """kernel analog filtering estimator"""
    def __init__(self, dim, kernel=None, approx_rank=None, eig_mult=1E-4) -> None:
        super().__init__()
        self.kaf = KAF(kernel=kernel, approx_rank=approx_rank, eig_mult=eig_mult)
        self.names = [f'x{i}' for i in range(dim)]

    def fit_next_step(self, X: np.ndarray, Y: np.ndarray, U: None) -> None:
        if U is not None:
            raise ValueError("KAF doesn't work for systems with inputs!")
        self.kaf.fit(X.T, Y.T)

    @property
    def model(self) -> asys.System:
        def step_func(t, x, i):
            return np.real(self.kaf.predict(np.atleast_2d(x)))
        return asys.StepDiscreteSystem(step_func, self.names)
