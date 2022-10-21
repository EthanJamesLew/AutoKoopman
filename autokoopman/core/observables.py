import abc
from typing import Sequence

import numpy as np
import sympy as sp  # type: ignore

from scipy.stats import cauchy, laplace


class KoopmanObservable(abc.ABC):
    """
    Koopman Observables Functions
       These objects implement the mapping of the system state to the Koopman invariant space (explicitly).

    References
        Explanation of Koopman Observables
           Brunton, S. L., & Kutz, J. N. (2022). Data-driven science and engineering: Machine learning, dynamical systems,
           and control. Cambridge University Press. pp 260-261
    """

    @abc.abstractmethod
    def obs_fcn(self, X: np.ndarray) -> np.ndarray:
        """
        Observables Function

        :param X: system states
        :returns: observables
        """
        pass

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.obs_fcn(X)

    def obs_grad(self, X: np.ndarray) -> np.ndarray:
        """
        Observables Gradient Function

        :param X: system states
        :returns: observables
        """
        raise NotImplementedError

    def __or__(self, obs: "KoopmanObservable"):
        """observable combination operator"""
        # TODO: implement combine
        return CombineObservable([self, obs])
        # identity | rffs


class CombineObservable(KoopmanObservable):
    def __init__(self, observables: Sequence[KoopmanObservable]):
        super(CombineObservable, self).__init__()
        self.observables = observables

    def obs_fcn(self, X: np.ndarray) -> np.ndarray:
        return np.vstack([obs.obs_fcn(X) for obs in self.observables])

    def obs_grad(self, X: np.ndarray) -> np.ndarray:
        return np.vstack([obs.obs_grad(X) for obs in self.observables])


class IdentityObservable(KoopmanObservable):
    def obs_fcn(self, X: np.ndarray) -> np.ndarray:
        return np.atleast_2d(X).T

    def obs_grad(self, X: np.ndarray) -> np.ndarray:
        assert len(X.shape) == 1
        return np.eye(len(X))


class SymbolicObservable(KoopmanObservable):
    def __init__(self, variables: Sequence[sp.Symbol], observables: Sequence[sp.Expr]):
        super(SymbolicObservable, self).__init__()
        self.length = len(variables)
        self._observables = observables
        self._variables = variables
        G = sp.Matrix(self._observables)
        GD = sp.Matrix([sp.diff(G, xi).T for xi in self._variables])
        self._g = sp.lambdify((self._variables,), G)
        self._gd = sp.lambdify((self._variables,), GD)

    def obs_fcn(self, X: np.ndarray) -> np.ndarray:
        return np.array(self._g(list(X.flatten())))

    def obs_grad(self, X: np.ndarray) -> np.ndarray:
        return np.array(self._gd(list(X))).T

    def __add__(self, obs: "SymbolicObservable"):
        return SymbolicObservable(
            list({*self._variables, *obs._variables}),
            [xi + yi for xi, yi in zip(self._observables, obs._observables)],
        )

    def __sub__(self, obs: "SymbolicObservable"):
        return SymbolicObservable(
            list({*self._variables, *obs._variables}),
            [xi - yi for xi, yi in zip(self._observables, obs._observables)],
        )

    def __mul__(self, obs: "SymbolicObservable"):
        return SymbolicObservable(
            list({*self._variables, *obs._variables}),
            [xi * yi for xi, yi in zip(self._observables, obs._observables)],
        )

    def __truediv__(self, obs: "SymbolicObservable"):
        return SymbolicObservable(
            list({*self._variables, *obs._variables}),
            [xi / yi for xi, yi in zip(self._observables, obs._observables)],
        )

    def __rdiv__(self, other):
        if isinstance(other, SymbolicObservable):
            return SymbolicObservable(
                list({*self._variables, *other._variables}),
                [xi / yi for xi, yi in zip(self._observables, other._observables)],
            )
        else:
            return SymbolicObservable(
                self._variables, [other / yi for yi in self._observables]
            )

    def __rmul__(self, other):
        if isinstance(other, SymbolicObservable):
            return SymbolicObservable(
                list({*self._variables, *other._variables}),
                [xi * yi for xi, yi in zip(self._observables, other._observables)],
            )
        else:
            return SymbolicObservable(
                self._variables, [other * yi for yi in self._observables]
            )

    def __or__(self, other):
        if isinstance(other, SymbolicObservable):
            return SymbolicObservable(
                list({*self._variables, *other._variables}),
                [*self._observables, *other._observables],
            )
        else:
            return CombineObservable([self, other])


class QuadraticObservable(SymbolicObservable):
    def __init__(self, length):
        """inefficient implementation to get quadratic koopman observables and its gradient functions"""
        vec = sp.symbols(" ".join([f"x{idx}" for idx in range(length)]))
        x = sp.Matrix((*vec, 1))
        U = x * x.T
        lv = [U[i, j] for i, j in zip(*np.tril_indices(len(x)))]
        super(QuadraticObservable, self).__init__(vec, lv)


class PolynomialObservable(KoopmanObservable):
    def __init__(self, dimension, degree) -> None:
        from sklearn.preprocessing import PolynomialFeatures

        super(PolynomialObservable, self).__init__()
        self.degree = degree
        self.dimension = dimension
        self.poly = PolynomialFeatures(int(self.degree), include_bias=False)
        self.poly.fit_transform(np.zeros((1, self.dimension)))

    def obs_fcn(self, X: np.ndarray) -> np.ndarray:
        return self.poly.transform(np.atleast_2d(X))


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
