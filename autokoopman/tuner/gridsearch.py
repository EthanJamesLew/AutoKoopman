import numpy as np

from autokoopman.core.trajectory import TrajectoriesData
from autokoopman.core.tuner import (
    TuneResults,
    TrajectoryScoring,
    HyperparameterMap,
    ParameterSpace,
)
import autokoopman.core.tuner as atuner
import itertools
from typing import Callable


class GridSearchTuner(atuner.HyperparameterTuner):
    @staticmethod
    def make_grid(space: ParameterSpace, n_samps):
        parameters = []
        for coord in space:
            if isinstance(coord, atuner.ContinuousParameter):
                if coord.distribution == "loguniform":
                    elems = np.logspace(
                        np.log10(coord._interval[0]),
                        np.log10(coord._interval[1]),
                        num=n_samps,
                    )
                    parameters.append(elems)
                elif coord.distribution:
                    parameters.append(
                        np.linspace(coord._interval[0], coord._interval[1], num=n_samps)
                    )
            elif isinstance(coord, atuner.FiniteParameter):
                parameters.append(list(coord.elements))
        return parameters

    def __init__(
        self,
        parameter_model: HyperparameterMap,
        training_data: TrajectoriesData,
        n_samps=10,
        **kwargs,
    ):
        super(GridSearchTuner, self).__init__(parameter_model, training_data, **kwargs)
        self.n_samps = n_samps

    def tune(
        self,
        nattempts=100,
        scoring_func: Callable[
            [TrajectoriesData, TrajectoriesData], float
        ] = TrajectoryScoring.end_point_score,
    ) -> TuneResults:
        # get the number of samples created from the meshgrid
        parameters = self.make_grid(self._parameter_model.parameter_space, self.n_samps)
        nsamples = np.prod(np.array([len(p) for p in parameters]))

        # create the sampler and send it the parameters
        sampling = self.tune_sampling(nsamples, scoring_func)
        next(sampling)

        for param in itertools.product(*parameters):
            try:
                sampling.send(param)
                next(sampling)
            except StopIteration:
                break
            except Exception as exc:
                print(f"Error: {exc}")
                self.error_messages.append((param, exc))
        return self.best_result
