import abc
from typing import Sequence

import numpy as np
import sympy as sp


class KoopmanObservable(abc.ABC):
    """explicit mapping from state to Koopman observables"""

    @abc.abstractmethod
    def obs_fcn(self, X: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.obs_fcn(X)

    def obs_grad(self, X: np.ndarray) -> np.ndarray:
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


class RFFObservable(KoopmanObservable):
    ...
