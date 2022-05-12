import abc
from typing import Callable, Sequence, Tuple, Optional, Union

import numpy as np
import scipy.integrate as scint  # type: ignore
import sympy as sp  # type: ignore

import autokoopman.core.trajectory as atraj
from autokoopman.core.format import _clip_list


class System(abc.ABC):
    """
    Dynamical System Base Class
        This is a base class for a dynamical system, which features:
            - A fixed dimensional state space (dimension) with names (names)
            - Initial Value Problem (IVP) the system can be simulated from initial conditions
    """
    @abc.abstractmethod
    def solve_ivp(
        self,
        initial_state: np.ndarray,
        tspan: Tuple[float, float],
        teval: Optional[np.ndarray] = None,
        sampling_period: float = 0.1,
    ) -> Union[atraj.Trajectory, atraj.UniformTimeTrajectory]:
        """
        Solve the initial value (IV) problem
            Given a system with a state space, specify how the system evolves with time given the initial conditions
            of the problem. The solution of a particular IV returns a trajectory over time teval if specified, or
            uniformly sampled over a time span given a sampling period.

        :param initial_state: IV state of system
        :param tspan: (if no teval is set) timespan to evolve system from iv (starting at tspan[0])
        :param teval: (optional) values of time sample the IVP trajectory
        :param sampling_period: (if no teval is set) sampling period of solution

        :returns: TimeTrajectory if teval is set, or UniformTimeTrajectory if not
        """
        raise NotImplementedError

    def solve_ivps(
        self,
        initial_states: np.ndarray,
        tspan: Tuple[float, float],
        teval: Optional[np.ndarray] = None,
        sampling_period: float = 0.1,
    ) -> Union[atraj.UniformTimeTrajectoriesData, atraj.TrajectoriesData]:
        ret = {}
        for idx, state in enumerate(initial_states):
            ret[idx] = self.solve_ivp(state, tspan, teval, sampling_period)
        return atraj.UniformTimeTrajectoriesData(ret) if teval is None else atraj.TrajectoriesData(ret)  # type: ignore

    @property
    @abc.abstractmethod
    def names(self) -> Sequence[str]:
        pass

    @property
    def dimension(self) -> int:
        return len(self.names)

    def __repr__(self):
        return f"<{self.__class__.__name__} Dimensions: {self.dimension} States: {_clip_list(self.names)}>"


class LinearSystem(System):
    """
    Linear Dynamical System
        There is a linear operator associated with the system.
    """
    @property
    @abc.abstractmethod
    def linear_operator(self):
        pass


class ContinuousSystem(System):
    """
    Continuous Time System
        In this case, a CT system is a system whose evolution function is defined by a gradient.
    """

    def solve_ivp(
        self,
        initial_state: np.ndarray,
        tspan: Tuple[float, float],
        teval: Optional[np.ndarray] = None,
        sampling_period: float = 0.1,
    ) -> Union[atraj.Trajectory, atraj.UniformTimeTrajectory]:
        """
        Solve the initial value (IV) problem for Continuous Time Systems
            Given a system with a state space, specify how the system evolves with time given the initial conditions
            of the problem. The solution of a particular IV returns a trajectory over time teval if specified, or
            uniformly sampled over a time span given a sampling period.

            A differential equation of the form :math:`\dot X_t = \operatorname{grad}(t, X_t)`.

        :param initial_state: IV state of system
        :param tspan: (if no teval is set) timespan to evolve system from iv (starting at tspan[0])
        :param teval: (optional) values of time sample the IVP trajectory
        :param sampling_period: (if no teval is set) sampling period of solution

        :returns: TimeTrajectory if teval is set, or UniformTimeTrajectory if not
        """
        if teval is None:
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
        else:
            sol = scint.solve_ivp(
                self.gradient,
                (min(teval), max(teval)),
                initial_state,
                # TODO: this is hacky
                t_eval=teval,
            )
            return atraj.Trajectory(sol.t, sol.y.T, self.names)

    def solve_ivps(
        self,
        initial_states: np.ndarray,
        tspan: Tuple[float, float],
        teval: Optional[np.ndarray] = None,
        sampling_period: float = 0.1,
    ) -> Union[atraj.UniformTimeTrajectoriesData, atraj.TrajectoriesData]:
        ret = {}
        for idx, state in enumerate(initial_states):
            ret[idx] = self.solve_ivp(state, tspan, teval, sampling_period)
        return atraj.UniformTimeTrajectoriesData(ret) if teval is None else atraj.TrajectoriesData(ret)  # type: ignore

    @abc.abstractmethod
    def gradient(self, time: float, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DiscreteSystem(System):
    """
    Discrete Time System
        In this case, a CT system is a system whose evolution function is defined by a next step function. For IVP, the
        discrete time can be related to continuous time via a sampling period. This trajectory can be interpolated to
        evaluate time points nonuniformly.
    """

    def solve_ivp(
        self,
        initial_state: np.ndarray,
        tspan: Tuple[float, float],
        teval: Optional[np.ndarray] = None,
        sampling_period: float = 0.1,
    ) -> Union[atraj.Trajectory, atraj.UniformTimeTrajectory]:
        """
        Solve the initial value (IV) problem for Discrete Time Systems
            Given a system with a state space, specify how the system evolves with time given the initial conditions
            of the problem. The solution of a particular IV returns a trajectory over time teval if specified, or
            uniformly sampled over a time span given a sampling period.

            A difference equation of the form :math:`X_{t+1} = \operatorname{step}(t, X_t)`.

        :param initial_state: IV state of system
        :param tspan: (if no teval is set) timespan to evolve system from iv (starting at tspan[0])
        :param teval: (optional) values of time sample the IVP trajectory
        :param sampling_period: (if no teval is set) sampling period of solution

        :returns: TimeTrajectory if teval is set, or UniformTimeTrajectory if not
        """
        if teval is None:
            times = np.arange(tspan[0], tspan[1] + sampling_period, sampling_period)
            states = np.zeros((len(times), len(self.names)))
            states[0] = np.array(initial_state).flatten()
            for idx, time in enumerate(times[1:]):
                states[idx + 1] = self.step(float(time), states[idx]).flatten()
            return atraj.UniformTimeTrajectory(
                states, sampling_period, self.names, tspan[0]
            )
        else:
            times = np.arange(min(teval), max(teval) + sampling_period, sampling_period)
            states = np.zeros((len(times), len(self.names)))
            states[0] = np.array(initial_state).flatten()
            for idx, time in enumerate(times[1:]):
                states[idx + 1] = self.step(float(time), states[idx]).flatten()
            traj = atraj.Trajectory(times, states, self.names)
            return traj.interp1d(teval)

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

    @property
    def latex(self):
        """
        System LaTeX
            Print the symbolic system as LaTeX (SymPy utility).
        """
        import sympy.printing as printing
        elem_lt = r'\\'.join([printing.latex(expr) for expr in self._exprs])
        return rf"\dot{{\mathbf x}} = \begin{{pmatrix}} {elem_lt} \end{{pmatrix}}"

    def display_math(self):
        """
        Jupyter Notebook Math Display
            Display the LaTeX math in a Jupyter notebook.
        """
        from IPython.display import display, Math, Markdown
        return display(Math(self.latex))


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


class LinearContinuousSystem(ContinuousSystem, LinearSystem):
    ...


class LinearDiscreteSystem(ContinuousSystem, LinearSystem):
    ...
