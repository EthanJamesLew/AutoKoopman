import abc

import numpy as np


class KoopmanObservable(abc.ABC):
    """explicit mapping from state to Koopman observables"""
    @abc.abstractmethod
    def obs_fcn(self, X: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.obs_fcn(X)

    def obs_grad(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __or__(self, obs: 'KoopmanObservable'):
        """observable combination operator"""
        # TODO: implement combine
        return CombineObservable([self, obs])
        # identity | rffs


class CombineObservable(KoopmanObservable):
    ...


class IdentityObservable(KoopmanObservable):
    ...


class SymbolicObservable(KoopmanObservable):
    ...


class RFFObservable(KoopmanObservable):
    ...