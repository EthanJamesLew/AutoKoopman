import abc
from typing import Tuple

import numpy as np

import autokoopman.trajectory as atraj


class ContinuousSystem(abc.ABC):
    """a continuous time system with defined gradient"""
    def solve_ivp(self, initial_state: np.ndarray,
                  tspan: Tuple[float, float],
                  sampling_period: float = 0.1) -> atraj.UniformTimeTrajectory:
        pass

    @abc.abstractmethod
    def gradient(self, time: float, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError


class SymbolicContinuousSystem(ContinuousSystem):
    ...


class GradientContinuousSystem(ContinuousSystem):
    ...


class LinearContinuousSystem(ContinuousSystem):
    ...


class KoopmanContinuousSystem(LinearContinuousSystem):
    ...
