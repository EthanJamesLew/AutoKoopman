import numpy as np

import autokoopman.core.estimator as aest
import autokoopman.core.system as asys


class KAF:
    """kernel anbalog filtering"""
    def __init__(self, kernel=None, approx_rank=None, eig_mult=1E-4) -> None:
        if kernel is None:
            self.kernel = RBF(X.shape[1])
        else:
            self.kernel = kernel
        self.approx_rank = approx_rank
        self.eig_mult = eig_mult

    def fit(self, X, Xp):
        self.X = X.T
        self.KXX = self.kernel.K(X, X)
        _lambda, self.subspace = np.linalg.eigs(self.KXX, k=self.approx_rank if self.approx_rank is not None else len(self.KXX)-1)
        self.eigvals = _lambda + max(_lambda) * self.eig_mult
        self.weights = ((Xp.T @ self.subspace) @ np.linalg.pinv(np.diag(self.eigvals))) @ self.subspace.conj().T
        
    def predict(self, Xtest):
        Xtest = Xtest.T
        KXY = self.kernel.K(self.X.T, Xtest.T)
        return (self.weights @ KXY).T


class KAFEstimator(aest.NextStepEstimator):
    def fit_next_step(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.kaf = KAF() 
        self.kaf.fit_multi(X, Y)

    @property
    def model(self) -> asys.System:
        def step_func(t, x):
            return np.real(self.kaf.predict(np.atleast_2d(x).T)).T
        return asys.StepDiscreteSystem(step_func, self.names)
