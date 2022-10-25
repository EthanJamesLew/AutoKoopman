import abc
from typing import Callable, Sequence, Tuple, Optional, Union

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
        tspan: Optional[Tuple[float, float]] = None,
        teval: Optional[np.ndarray] = None,
        inputs: Optional[np.ndarray] = None,
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
        tspan: Optional[Tuple[float, float]] = None,
        teval: Optional[np.ndarray] = None,
        inputs: Optional[np.ndarray] = None,
        sampling_period: float = 0.1,
    ) -> Union[atraj.UniformTimeTrajectoriesData, atraj.TrajectoriesData]:
        ret = {}
        for idx, state in enumerate(initial_states):
            ret[idx] = self.solve_ivp(
                state,
                tspan=tspan,
                teval=teval,
                inputs=inputs[idx] if inputs is not None else None,
                sampling_period=sampling_period,
            )
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


class ContinuousSystem(System):
    """
    Continuous Time System
        In this case, a CT system is a system whose evolution function is defined by a gradient.
    """

    def solve_ivp(
        self,
        initial_state: np.ndarray,
        tspan: Optional[Tuple[float, float]] = None,
        teval: Optional[np.ndarray] = None,
        inputs: Optional[np.ndarray] = None,
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
        if teval is None and tspan is None:
            raise RuntimeError(f"teval or tspan must be set")
        if inputs is None:
            if teval is None:
                t_eval = np.arange(
                    tspan[0], tspan[-1] + sampling_period * (1 - 1e-12), sampling_period
                )
                sol = scint.solve_ivp(
                    self.gradient,
                    (min(t_eval), max(t_eval)),
                    initial_state,
                    args=(None,),
                    t_eval=t_eval,
                )
                return atraj.UniformTimeTrajectory(
                    sol.y.T,
                    None,
                    sampling_period,
                    state_names=self.names,
                    start_time=tspan[0],
                )
            else:
                sol = scint.solve_ivp(
                    self.gradient,
                    (min(teval), max(teval)),
                    initial_state,
                    args=(None,),
                    # TODO: this is hacky
                    t_eval=teval,
                )
                return atraj.Trajectory(sol.t, sol.y.T, None, self.names)
        else:
            if teval is not None:
                if len(teval) == 0:
                    raise ValueError("teval must have at least one value")
                inputs = np.array(inputs)
                sol = [initial_state]
                if len(teval) > 1:
                    for tcurrent, tnext, inpi in zip(
                        teval[:-1], teval[1:], inputs[:-1]
                    ):
                        sol_next = scint.solve_ivp(
                            self.gradient,
                            (tcurrent, tnext),
                            sol[-1],
                            args=(np.atleast_1d(inpi),),
                            t_eval=(tcurrent, tnext),
                        )
                        sol.append(sol_next.y.T[-1])
                if len(inputs.shape) == 1:
                    inputs = inputs[:, np.newaxis]
                return atraj.Trajectory(
                    np.array(teval), np.array(sol), inputs, self.names
                )
            else:
                raise RuntimeError("teval must be set if inputs is set")

    def solve_ivps(
        self,
        initial_states: np.ndarray,
        tspan: Optional[Tuple[float, float]] = None,
        teval: Optional[np.ndarray] = None,
        inputs: Optional[np.ndarray] = None,
        sampling_period: float = 0.1,
    ) -> Union[atraj.UniformTimeTrajectoriesData, atraj.TrajectoriesData]:
        ret = {}
        if inputs is not None:
            assert len(inputs) == len(
                initial_states
            ), f"length of inputs {len(inputs)} must match length of initial states {len(initial_states)}"
        for idx, state in enumerate(initial_states):
            ret[idx] = self.solve_ivp(
                state,
                tspan=tspan,
                teval=teval,
                inputs=inputs[idx] if inputs is not None else None,
                sampling_period=sampling_period,
            )
        return atraj.UniformTimeTrajectoriesData(ret) if teval is None else atraj.TrajectoriesData(ret)  # type: ignore

    @abc.abstractmethod
    def gradient(
        self, time: float, state: np.ndarray, sinput: Optional[np.ndarray]
    ) -> np.ndarray:
        raise NotImplementedError


class DiscreteSystem(System):
    """
    Discrete Time System
        In this case, a CT system is a system whose evolution function is defined by a next step function. For IVP, the
        discrete time can be related to continuous time via a sampling period. This trajectory can be interpolated to
        evaluate time points nonuniformly.

        TODO: should this have a sampling period instance member?
    """

    def solve_ivp(
        self,
        initial_state: np.ndarray,
        tspan: Optional[Tuple[float, float]] = None,
        teval: Optional[np.ndarray] = None,
        inputs: Optional[np.ndarray] = None,
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
        if teval is None and tspan is None:
            raise RuntimeError(f"teval or tspan must be set")
        if inputs is None:
            if teval is None:
                times = np.arange(tspan[0], tspan[1] + sampling_period, sampling_period)
                states = np.zeros((len(times), len(self.names)))
                states[0] = np.array(initial_state).flatten()
                for idx, time in enumerate(times[1:]):
                    states[idx + 1] = self.step(
                        float(time), states[idx], None
                    ).flatten()
                return atraj.UniformTimeTrajectory(
                    states,
                    None,
                    sampling_period,
                    state_names=self.names,
                    start_time=tspan[0],
                )
            else:
                times = np.arange(
                    min(teval), max(teval) + sampling_period, sampling_period
                )
                states = np.zeros((len(times), len(self.names)))
                states[0] = np.array(initial_state).flatten()
                for idx, time in enumerate(times[1:]):
                    states[idx + 1] = self.step(
                        float(time), states[idx], None
                    ).flatten()
                traj = atraj.Trajectory(times, states, None, self.names)
                return traj.interp1d(teval)
        else:
            if teval is not None:
                if len(teval) == 0:
                    raise ValueError("teval must have at least one value")
                if inputs.ndim == 1:
                    inputs = inputs[:, np.newaxis]
                teval = np.array(teval)
                times = np.arange(
                    min(teval), max(teval) + sampling_period, sampling_period
                )
                states = np.zeros((len(times), len(self.names)))
                states[0] = np.array(initial_state).flatten()
                for idx, time in enumerate(times[1:]):
                    diff = time - teval
                    diff[diff < 0.0] = float("inf")
                    tidx = diff.argmin()
                    states[idx + 1] = self.step(
                        float(time), states[idx], np.atleast_1d(inputs[tidx])
                    ).flatten()
                traj = atraj.Trajectory(times, states, None, state_names=self.names)
                ctraj = traj.interp1d(teval)
                return atraj.Trajectory(teval, ctraj.states, inputs, self.names)
            else:
                raise RuntimeError("teval must be set if inputs is set")

    @abc.abstractmethod
    def step(
        self, time: float, state: np.ndarray, sinput: Optional[np.ndarray]
    ) -> np.ndarray:
        raise NotImplementedError


class SymbolicContinuousSystem(ContinuousSystem):
    def __init__(
        self,
        variables: Sequence[sp.Symbol],
        gradient_exprs: Sequence[sp.Expr],
        input_variables: Optional[Sequence[sp.Symbol]] = None,
        time_var=None,
    ):
        if time_var is None:
            time_var = sp.symbols("_t0")
        if input_variables is None:
            self._variables = [time_var, *variables]
        else:
            self._variables = [time_var, *variables, *input_variables]
        self._state_vars = variables
        self._input_vars = input_variables
        self._exprs = gradient_exprs
        self._mat = sp.Matrix(self._exprs)
        self._fmat = sp.lambdify((self._variables,), self._mat)

    def gradient(
        self, time: float, state: np.ndarray, sinput: Optional[np.ndarray]
    ) -> np.ndarray:
        if sinput is None:
            return np.array(self._fmat(np.array([time, *state]))).flatten()
        else:
            return np.array(self._fmat(np.array([time, *state, *sinput]))).flatten()

    @property
    def names(self) -> Sequence[str]:
        return [str(s) for s in self._state_vars]


class GradientContinuousSystem(ContinuousSystem):
    def __init__(
        self,
        gradient_func: Callable[[float, np.ndarray, Optional[np.ndarray]], np.ndarray],
        names,
    ):
        self._names = names
        self._gradient_func = gradient_func

    def gradient(
        self, time: float, state: np.ndarray, sinput: Optional[np.ndarray]
    ) -> np.ndarray:
        return self._gradient_func(time, state, sinput)

    @property
    def names(self):
        return self._names


class StepDiscreteSystem(DiscreteSystem):
    def __init__(
        self,
        step_func: Callable[[float, np.ndarray, Optional[np.ndarray]], np.ndarray],
        names,
    ):
        self._names = names
        self._step_func = step_func

    def step(
        self, time: float, state: np.ndarray, sinput: Optional[np.ndarray]
    ) -> np.ndarray:
        return self._step_func(time, state, sinput)

    @property
    def names(self):
        return self._names


class LinearContinuousSystem(ContinuousSystem):
    ...


class KoopmanContinuousSystem(LinearContinuousSystem):
    ...
