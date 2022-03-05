import abc
from typing import Tuple, Sequence

import numpy as np
import sympy as sp
import scipy.integrate as scint

import autokoopman.trajectory as atraj
from autokoopman.format import _clip_list


class ContinuousSystem(abc.ABC):
    """a continuous time system with defined gradient"""
    def solve_ivp(self,
                  initial_state: np.ndarray,
                  tspan: Tuple[float, float],
                  sampling_period: float = 0.1) -> atraj.UniformTimeTrajectory:
        sol = scint.solve_ivp(self.gradient,
                              tspan,
                              initial_state,
                              t_eval=np.arange(0, tspan[-1] + sampling_period, sampling_period))
        return atraj.UniformTimeTrajectory(sol.y.T, sampling_period, self.names, tspan[0])

    def solve_ivps(self,
                  initial_states: np.ndarray,
                  tspan: Tuple[float, float],
                  sampling_period: float = 0.1) -> atraj.UniformTimeTrajectoriesData:
        ret = {}
        for idx, state in enumerate(initial_states):
            ret[idx] = self.solve_ivp(state, tspan, sampling_period)
        return atraj.UniformTimeTrajectoriesData(ret)

    @abc.abstractmethod
    def gradient(self, time: float, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def names(self) -> Sequence[str]:
        pass

    @property
    def dimension(self) -> int:
        return len(self.names)

    def __repr__(self):
        return f"<{self.__class__.__name__} Dimensions: {self.dimension} States: {_clip_list(self.names)}>"


class SymbolicContinuousSystem(ContinuousSystem):
    def __init__(self, variables: Sequence[sp.Symbol],
                 gradient_exprs: Sequence[sp.Expr],
                 time_var = None):
        if time_var is None:
            time_var = sp.symbols('_t0')
        self._variables = [time_var, *variables]
        self._state_vars = variables
        self._exprs = gradient_exprs
        self._mat = sp.Matrix(self._exprs)
        self._fmat = sp.lambdify((self._variables,), self._mat)

    def gradient(self, time: float, state: np.ndarray) -> np.ndarray:
        return np.array(self._fmat(np.array([time, *state]))).flatten()

    @property
    def names(self) -> Sequence[str]:
        return [str(s) for s in self._state_vars]


class GradientContinuousSystem(ContinuousSystem):
    ...


class LinearContinuousSystem(ContinuousSystem):
    ...


class KoopmanContinuousSystem(LinearContinuousSystem):
    ...
