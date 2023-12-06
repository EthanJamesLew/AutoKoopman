from typing import Optional
import numpy as np


class OnlineInputDMD:
    """Online Dynamic Mode Decomposition with Inputs

    This is a variation on Zhang et al. with extensions for KIC
    """

    def __init__(self, n: int, m: Optional[int], weighting=0.9) -> None:
        self.n = n
        self.m = m if m is not None else 0

        self.weighting = weighting
        self.timestep = 0
        self.A = np.zeros((self.n, self.n + self.m))
        self._P = np.zeros((self.n + self.m, self.n + self.m))
        self._initialize()
        self.ready = False

    def _initialize(self) -> None:
        epsilon = 1e-15
        alpha = 1.0 / epsilon
        self.A = np.random.randn(self.n, self.n)
        self._P = alpha * np.identity(self.n)

    def initialize(
        self, Xp: np.ndarray, Yp: np.ndarray, Up: Optional[np.ndarray]
    ) -> None:
        assert Xp is not None and Yp is not None
        Xp, Yp, Up = (
            np.array(Xp),
            np.array(Yp),
            np.array(Up) if Up is not None else None,
        )
        assert Xp.shape == Yp.shape
        assert Xp.shape[0] == self.n

        if Up is not None:
            assert Up.shape[0] == self.m

        # necessary condition for over-constrained initialization
        p = Xp.shape[1]
        assert (
            p >= self.n and np.linalg.matrix_rank(Xp) == self.n
        ), "WARNING: initialization is underconstrained"

        # for KIC -- treat inputs as state
        if self.m > 0:
            Zp = np.vstack((Xp, Up))
        else:
            Zp = Xp

        weight = np.sqrt(self.weighting) ** range(p - 1, -1, -1)
        Zqhat, Yqhat = weight * Zp, weight * Yp
        self.A = Yqhat.dot(np.linalg.pinv(Zqhat))
        self._P = np.linalg.inv(Zqhat.dot(Zqhat.T)) / self.weighting
        self.timestep += p

        if self.timestep >= 2 * self.n:
            self.ready = True

    def update(self, x: np.ndarray, y: np.ndarray, u: Optional[np.ndarray]) -> None:
        assert x is not None and y is not None
        x, y = np.array(x), np.array(y)
        assert np.array(x).shape == np.array(y).shape
        assert np.array(x).shape[0] == self.n
        if u is not None:
            assert np.array(u).shape[0] == self.m

        # for KIC -- treat inputs as state
        if self.m > 0:
            z = np.hstack((x, u))
        else:
            z = x

        # compute P*x matrix vector product beforehand
        Pz = self._P.dot(z)
        # compute gamma
        gamma = 1.0 / (1 + z.T.dot(Pz))
        # update A
        self.A += np.outer(gamma * (y - self.A.dot(z)), Pz)
        # update P, group Px*Px' to ensure positive definite
        self._P = (self._P - gamma * np.outer(Pz, Pz)) / self.weighting
        # ensure P is SPD by taking its symmetric part
        self._P = (self._P + self._P.T) / 2

        # time step + 1
        self.timestep += 1

        if self.timestep >= 2 * self.n:
            self.ready = True
