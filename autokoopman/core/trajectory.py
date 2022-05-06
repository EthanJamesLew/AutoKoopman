from typing import Dict, Hashable, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from autokoopman.core.format import _clip_list


class Trajectory:
    """a trajectory is a time series of vectors"""

    def __init__(
        self,
        times: np.ndarray,
        states: np.ndarray,
        state_names: Sequence[str],
        threshold=None,
    ):
        assert times.ndim == 1, "times must be a 1d array"
        self._times = times
        self._states = states
        self._state_names = state_names
        self._threshold = np.finfo(times.dtype).eps if threshold is None else threshold
        assert self.dimension == len(
            self.names
        ), "dimension must match the length of state names!"

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def size(self) -> int:
        return self.states.shape[0]

    @property
    def dimension(self) -> int:
        return self.states.shape[1]

    @property
    def states(self) -> np.ndarray:
        return self._states

    @property
    def names(self) -> Sequence:
        return self._state_names

    @property
    def is_uniform_time(self) -> bool:
        """determine if a trajectory is uniform time"""
        # FIXME: len greater than 3??
        time_steps = np.diff(self._times)
        dtimes = np.diff(time_steps)
        return bool(np.all([ti < self._threshold for ti in np.abs(dtimes)]))

    def interp1d(self, times) -> "Trajectory":
        from scipy.interpolate import interp1d  # type: ignore

        times = np.array(times)
        f = interp1d(
            np.array(self._times),
            np.array(self._states),
            fill_value="extrapolate",
            axis=0,
        )
        states = f(times)
        return Trajectory(times, states, self.names, threshold=self._threshold)

    def interp_uniform_time(self, sampling_period) -> "UniformTimeTrajectory":
        ts = np.arange(
            np.min(self._times), np.max(self._times) + sampling_period, sampling_period
        )
        return self.interp1d(ts)

    def to_uniform_time_traj(self) -> "UniformTimeTrajectory":
        assert self.is_uniform_time, "trajectory must be uniform time to apply"
        sp = np.diff(self._times)[0]
        start = self._times[0]
        return UniformTimeTrajectory(
            self.states, sp, self.names, start_time=start, threshold=self._threshold
        )

    def __repr__(self):
        return f"<{self.__class__.__name__} Dim: {self.dimension}, Length: {self.size}, State Names: {_clip_list(self.names)}>"

    def _prepare_other(self, other: "Trajectory"):
        if len(self.times) != len(other.times) or not np.all(
            np.isclose(self.times, other.times)
        ):
            return other.interp1d(self.times)
        else:
            return other

    def __sub__(self, other: "Trajectory") -> "Trajectory":
        otheri = self._prepare_other(other)
        return Trajectory(
            self.times,
            self.states - otheri.states,
            self.names,
            threshold=self._threshold,
        )

    def __add__(self, other: "Trajectory") -> "Trajectory":
        otheri = self._prepare_other(other)
        return Trajectory(
            self.times,
            self.states + otheri.states,
            self.names,
            threshold=self._threshold,
        )

    def norm(self, axis=1) -> "Trajectory":
        from numpy.linalg import norm

        return Trajectory(
            self.times,
            norm(self.states, axis=axis)[:, np.newaxis],
            ["<norm>"],
            threshold=self._threshold,
        )


class UniformTimeTrajectory(Trajectory):
    """uniform time is a trajectory advanced by a sampling period"""

    def __init__(
        self,
        states: np.ndarray,
        sampling_period: float,
        state_names: Sequence[str],
        start_time=0.0,
        threshold=None,
    ):
        nsteps = len(states)
        times = np.arange(
            start_time, start_time + nsteps * sampling_period, sampling_period
        )[:nsteps]
        super().__init__(times, states, state_names, threshold=threshold)
        self._sampling_period = sampling_period

    @property
    def sampling_period(self) -> float:
        return self._sampling_period

    @property
    def sampling_frequency(self) -> float:
        return 1.0 / self.sampling_period

    @property
    def is_uniform_time(self) -> bool:
        """determine if a trajectory is uniform time"""
        return True


class TrajectoriesData:
    """a dataset of trajectories"""

    traj_id_hname = "id"
    time_id_hname = "time"

    @staticmethod
    def equal_lists(lists):
        return not lists or all(lists[0] == b for b in lists[1:])

    @classmethod
    def from_pandas(cls, data_df: pd.DataFrame, threshold=None):
        traj_ids = set(data_df[cls.traj_id_hname])
        assert cls.traj_id_hname in set(data_df.columns) and cls.time_id_hname in set(
            data_df.columns
        ), (f"csv must have {cls.traj_id_hname} " f"and {cls.time_id_hname} fields")
        assert len(data_df.columns) > 2, "csv has to have more than two columns"
        tidx, sidx = (
            list(data_df.columns).index(cls.time_id_hname),
            list(data_df.columns).index(cls.traj_id_hname),
        )
        state_names = list(data_df.columns)[tidx + 1 : sidx]
        trajs = {
            uvi: Trajectory(
                data_df[data_df[cls.traj_id_hname] == uvi][
                    cls.time_id_hname
                ].to_numpy(),
                data_df[data_df[cls.traj_id_hname] == uvi][state_names].to_numpy(),
                state_names,
                threshold=threshold,
            )
            for uvi in traj_ids
        }
        return cls(trajs)

    @classmethod
    def from_csv(cls, fname: str, threshold=None):
        """csv deserialization"""
        data_df = pd.read_csv(fname)
        return cls.from_pandas(data_df, threshold=threshold)

    def to_pandas(self):
        # form the dataframe
        serial_data = np.vstack(
            [
                np.hstack(
                    [
                        traji.times[:, np.newaxis],
                        traji.states,
                        np.array([namei] * len(traji.times), dtype=type(namei))[
                            :, np.newaxis
                        ],
                    ]
                )
                for idx, (namei, traji) in enumerate(self._trajs.items())
            ]
        )
        columns = [self.time_id_hname, *self.state_names, self.traj_id_hname]
        return pd.DataFrame(columns=columns, data=serial_data)

    def to_csv(self, fname: str):
        """csv serialization"""
        df = self.to_pandas()
        # write to csv
        df.to_csv(fname, index=False)

    def interp_uniform_time(self, sampling_period) -> "UniformTimeTrajectoriesData":
        return UniformTimeTrajectoriesData(
            {k: v.interp_uniform_time(sampling_period) for k, v in self._trajs.items()}
        )

    def __init__(self, trajs: Dict[Hashable, Union[UniformTimeTrajectory, Trajectory]]):
        self._trajs: Dict[Hashable, Union[UniformTimeTrajectory, Trajectory]] = trajs
        self._names_list = [v.names for _, v in self._trajs.items()]
        assert self.equal_lists(self._names_list), "all state names must be the same"
        self._state_names = self._names_list[0]

    def get_trajectory(self, traj_id: Hashable) -> Trajectory:
        return self._trajs[traj_id]

    @property
    def n_trajs(self) -> int:
        return len(self._trajs)

    @property
    def state_names(self) -> Sequence[str]:
        return self._state_names

    @property
    def traj_names(self) -> Set[Hashable]:
        return set(self._trajs.keys())

    def __getitem__(self, item):
        return self.get_trajectory(item)

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} N Trajs: {self.n_trajs}, Traj "
            f"Names: [{_clip_list(list(self.traj_names))}], State "
            f"Names: [{_clip_list(self.state_names)}]>"
        )

    def __iter__(self):
        for _, v in self._trajs.items():
            yield v

    def __len__(self):
        return len(self._trajs)

    def _prepare_other(self, other: "TrajectoriesData"):
        assert set(other.traj_names) == set(
            self.traj_names
        ), "other data traj names must have same names"
        return other

    def __sub__(self, other: "TrajectoriesData"):
        otheri = self._prepare_other(other)
        return TrajectoriesData({k: v - otheri[k] for k, v in self._trajs.items()})

    def __add__(self, other: "TrajectoriesData"):
        otheri = self._prepare_other(other)
        return TrajectoriesData({k: v + otheri[k] for k, v in self._trajs.items()})

    def norm(self):
        return TrajectoriesData({k: v.norm() for k, v in self._trajs.items()})


class UniformTimeTrajectoriesData(TrajectoriesData):
    """a dataset of uniform time trajectories"""

    def __init__(self, trajs: Dict[Hashable, Union[UniformTimeTrajectory, Trajectory]]):
        super(UniformTimeTrajectoriesData, self).__init__(trajs)

    @property
    def sampling_period(self) -> float:
        return [
            v.sampling_period
            for _, v in self._trajs.items()
            if isinstance(v, UniformTimeTrajectory)
        ][0]

    @classmethod
    def from_pandas(cls, data_df: pd.DataFrame, threshold=None):
        """csv deserialization"""
        traj_data = super().from_pandas(data_df, threshold=threshold)
        for k, v in traj_data._trajs.items():
            assert v.is_uniform_time, f"{k} is not uniform time"
        return cls({k: v.to_uniform_time_traj() for k, v in traj_data._trajs.items()})

    @property
    def next_step_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        X = np.vstack([x.states[:-1, :] for _, x in self._trajs.items()]).T

        Xp = np.vstack([x.states[1:, :] for _, x in self._trajs.items()]).T

        return X, Xp
