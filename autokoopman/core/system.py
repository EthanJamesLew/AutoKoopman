import abc
from typing import Callable, Sequence, Tuple

import numpy as np
import scipy.integrate as scint  # type: ignore
import sympy as sp  # type: ignore

import autokoopman.core.trajectory as atraj
from autokoopman.core.format import _clip_list


class System(abc.ABC):
    @abc.abstractmethod
    def solve_ivp(
        self,
        initial_state: np.ndarray,
        tspan: Tuple[float, float],
        sampling_period: float = 0.1,
    ) -> atraj.UniformTimeTrajectory:
        raise NotImplementedError

    def solve_ivps(
        self,
        initial_states: np.ndarray,
        tspan: Tuple[float, float],
        sampling_period: float = 0.1,
    ) -> atraj.UniformTimeTrajectoriesData:
        ret = {}
        for idx, state in enumerate(initial_states):
            ret[idx] = self.solve_ivp(state, tspan, sampling_period)
        return atraj.UniformTimeTrajectoriesData(ret)  # type: ignore

    @property
    @abc.abstractmethod
    def names(self) -> Sequence[str]:
        pass

    @property
    def dimension(self) -> int:
        return len(self.names)

    def __repr__(self):
        return f"<{self.__class__.__name__} Dimensions: {self.dimension} States: {_clip_list(self.names)}>"


class ContinuousSystem(System):
    """a continuous time system with defined gradient"""

    def solve_ivp(
        self,
        initial_state: np.ndarray,
        tspan: Tuple[float, float],
        sampling_period: float = 0.1,
    ) -> atraj.UniformTimeTrajectory:
        sol = scint.solve_ivp(
            self.gradient,
            tspan,
            initial_state,
            # TODO: this is hacky
            t_eval=np.arange(
                tspan[0], tspan[-1] + sampling_period - 1e-10, sampling_period
            ),
        )
        return atraj.UniformTimeTrajectory(
            sol.y.T, sampling_period, self.names, tspan[0]
        )

    def solve_ivps(
        self,
        initial_states: np.ndarray,
        tspan: Tuple[float, float],
        sampling_period: float = 0.1,
    ) -> atraj.UniformTimeTrajectoriesData:
        ret = {}
        for idx, state in enumerate(initial_states):
            ret[idx] = self.solve_ivp(state, tspan, sampling_period)
        return atraj.UniformTimeTrajectoriesData(ret)  # type: ignore

    @abc.abstractmethod
    def gradient(self, time: float, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DiscreteSystem(System):
    def solve_ivp(
        self,
        initial_state: np.ndarray,
        tspan: Tuple[float, float],
        sampling_period: float = 0.1,
    ) -> atraj.UniformTimeTrajectory:
        times = np.arange(tspan[0], tspan[1] + sampling_period, sampling_period)
        states = np.zeros((len(times), len(self.names)))
        states[0] = np.array(initial_state).flatten()
        for idx, time in enumerate(times[1:]):
            states[idx + 1] = self.step(float(time), states[idx]).flatten()
        return atraj.UniformTimeTrajectory(
            states, sampling_period, self.names, tspan[0]
        )

    @abc.abstractmethod
    def step(self, time: float, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SymbolicContinuousSystem(ContinuousSystem):
    def __init__(
        self,
        variables: Sequence[sp.Symbol],
        gradient_exprs: Sequence[sp.Expr],
        time_var=None,
    ):
        if time_var is None:
            time_var = sp.symbols("_t0")
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
    def __init__(self, gradient_func: Callable[[float, np.ndarray], np.ndarray], names):
        self._names = names
        self._gradient_func = gradient_func

    def gradient(self, time: float, state: np.ndarray) -> np.ndarray:
        return self._gradient_func(time, state)

    @property
    def names(self):
        return self._names


class StepDiscreteSystem(DiscreteSystem):
    def __init__(self, step_func: Callable[[float, np.ndarray], np.ndarray], names):
        self._names = names
        self._step_func = step_func

    def step(self, time: float, state: np.ndarray) -> np.ndarray:
        return self._step_func(time, state)

    @property
    def names(self):
        return self._names


class LinearContinuousSystem(ContinuousSystem):
    ...


class KoopmanContinuousSystem(LinearContinuousSystem):
    ...
