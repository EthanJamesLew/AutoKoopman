import autokoopman.core.estimator as kest
import autokoopman.core.system as ksys
import numpy as np
from scipy.linalg import logm
import functools
import autokoopman.core.observables as kobs


def requires_trained(f):
    @functools.wraps(f)
    def inner(estimator, *args, **kwargs):
        if estimator.has_trained:
            return f(estimator, *args, **kwargs)
        else:
            raise RuntimeError(
                f"method {f} requires that the estimator {estimator} has been trained"
            )

    return inner


class Estimator:
    """TODO: FIXME: maybe replace with scikit estimator"""

    def __init__(self):
        self.has_trained = False

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class KoopmanSystemEstimator(Estimator):
    def __init__(
        self, observable_fcn: kobs.KoopmanObservable, sampling_period: float, rank=10
    ):
        super().__init__()
        self.obs = observable_fcn
        self.g = observable_fcn.obs_fcn
        self.gd = observable_fcn.obs_grad
        self.rank = rank
        self.sampling_period = sampling_period
        self._X, self._Xn = None, None
        self._system, self._A, self._B = None, None, None

    @requires_trained
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array(
            [
                self._system.trajectory(
                    x, (0.0, self.sampling_period), self.sampling_period
                )[-1]
                for x in X
            ]
        )


def dmd(X, Xp, r=2):
    """simple dynamic mode decomposition (dmd) implementation"""
    # compute Atilde
    U, Sigma, V = np.linalg.svd(X, False)
    U, Sigma, V = U[:, :r], np.diag(Sigma)[:r, :r], V.conj().T[:, :r]
    Atilde = U.conj().T @ Xp @ V @ np.linalg.inv(Sigma)

    # eigenvalues of Atilde are same as A
    D, W = np.linalg.eig(Atilde)

    # recover U.shape[0] eigenfunctions of A
    phi = Xp @ V @ (np.linalg.inv(Sigma)) @ W
    return D, phi


def evolve_modes(mu, Phi, ic, sampling_period, tmax=20.0):
    """time evolve modes of the DMD"""
    b = np.linalg.pinv(Phi) @ np.array(ic)
    t = np.linspace(0, tmax, int(tmax / sampling_period))
    dt = t[2] - t[1]
    psi = np.zeros([len(ic), len(t)], dtype="complex")
    for i, ti in enumerate(t):
        psi[:, i] = np.multiply(np.power(mu, ti / dt), b)
    return psi


def learned_sys(Xc, Xcp, p, g, gradg, sampling_period):
    Yc, Ycp = np.hstack([g(x, p=p) for x in Xc.T]), np.hstack(
        [g(x, p=p) for x in Xcp.T]
    )
    A = Ycp @ np.linalg.pinv(Yc)
    B = logm(A) / sampling_period

    def sys(t, x):
        return np.real(np.linalg.pinv(gradg(x, p=p)) @ B @ g(x, p=p)).flatten()

    return sys


def learned_dmd_sys(Xc, Xcp, g, gradg, sampling_period, r=30):
    Yc, Ycp = np.hstack([g(x) for x in Xc.T]), np.hstack([g(x) for x in Xcp.T])
    mucy, Phicy = dmd(Yc, Ycp, r=r)
    A = Phicy @ np.diag(mucy) @ np.linalg.pinv(Phicy)
    B = logm(A) / sampling_period

    def sys(t, x):
        return np.real(np.linalg.pinv(gradg(x)) @ B @ g(x)).flatten()

    return sys


def get_linear_transform(Xc, Xcp, g, sampling_period, r=30, continuous=True):
    Yc, Ycp = np.hstack([g(x) for x in Xc.T]), np.hstack([g(x) for x in Xcp.T])
    mucy, Phicy = dmd(Yc, Ycp, r)
    A = Phicy @ np.diag(mucy) @ np.linalg.pinv(Phicy)
    if continuous:
        B = logm(A) / sampling_period
        return B
    else:
        return A


class KoopmanDiscSystemEstimator(KoopmanSystemEstimator):
    def fit_disc(self, X: np.ndarray, Xn: np.ndarray) -> None:
        Yc, Ycp = np.hstack([self.g(x) for x in X.T]), np.hstack(
            [self.g(x) for x in Xn.T]
        )
        mucy, Phicy = dmd(Yc, Ycp, r=self.rank)
        self._A = Phicy @ np.diag(mucy) @ np.linalg.pinv(Phicy)

    def predict(self, x):
        return np.real(self._A @ np.atleast_2d(self.g(x))).flatten()


class KoopmanDiscEstimator(kest.NextStepEstimator):
    def __init__(self, observables, sampling_period, dim, rank):
        self._est = KoopmanDiscSystemEstimator(observables, sampling_period, rank=rank)
        self.dim = dim

    def fit_next_step(self, X: np.ndarray, Y: np.ndarray) -> None:
        self._est.fit_disc(X, Y)

    @property
    def model(self) -> ksys.System:
        def step_func(t, x):
            return np.real(self._est.predict(np.atleast_2d(x))[:2])

        return ksys.StepDiscreteSystem(step_func, self.names)
