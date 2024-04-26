import numpy as np
from scipy.stats import cauchy, laplace

from .observables import KoopmanObservable


class RFFObservable(KoopmanObservable):
    def __init__(self, dimension, num_features, gamma, metric="rbf"):
        super(RFFObservable, self).__init__()
        self.gamma = gamma
        self.dimension = dimension
        self.metric = metric
        self.D = num_features
        # Generate D iid samples from p(w)
        if self.metric == "rbf":
            self.w = np.sqrt(2 * self.gamma) * np.random.normal(
                size=(self.D, self.dimension)
            )
        elif self.metric == "laplace":
            self.w = cauchy.rvs(scale=self.gamma, size=(self.D, self.dimension))
        # Generate D iid samples from Uniform(0,2*pi)
        self.u = 2 * np.pi * np.random.rand(self.D)

    def obs_fcn(self, X: np.ndarray) -> np.ndarray:
        # modification...
        if len(X.shape) == 1:
            x = np.atleast_2d(X.flatten()).T
        else:
            x = X.T
        w = self.w.T
        u = self.u[np.newaxis, :].T
        s = np.sqrt(2 / self.D)
        Z = s * np.cos(x.T @ w + u.T)
        return Z.T

    def obs_grad(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 1:
            x = np.atleast_2d(X.flatten()).T
        else:
            x = X.T
        x = np.atleast_2d(X.flatten()).T
        w = self.w.T
        u = self.u[np.newaxis, :].T
        s = np.sqrt(2 / self.D)
        # TODO: make this sparse?
        Z = -s * np.diag(np.sin(u + w.T @ x).flatten()) @ w.T
        return Z

    def compute_kernel(self, X: np.ndarray) -> np.ndarray:
        Z = self.transform(X)
        K = Z.dot(Z.T)
        return K