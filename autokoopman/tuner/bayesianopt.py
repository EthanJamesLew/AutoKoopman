import numpy as np

from autokoopman.core.trajectory import TrajectoriesData
from autokoopman.core.tuner import (
    TuneResults,
    TrajectoryScoring,
    HyperparameterMap,
    ParameterSpace,
    FiniteParameter,
    DiscreteParameter,
    ContinuousParameter,
)
import autokoopman.core.tuner as atuner
from typing import Callable


class BayesianOptTuner(atuner.HyperparameterTuner):
    r"""Bayesian Optimization Hyperparameter Tuner"""

    @staticmethod
    def make_bounds(param_space: ParameterSpace):
        """make GPyOpt Optimization Problem domain"""
        bounds = []
        lengths = []
        for idx, coord in enumerate(param_space):
            if isinstance(coord, ContinuousParameter):
                g_var = {
                    "name": f"var_{coord.name}_{idx}",
                    "type": "continuous"
                    if isinstance(coord, ContinuousParameter)
                    else "discrete",
                    "domain": (coord._interval[0], coord._interval[1]),
                }
                g_len = coord._interval[1] - coord._interval[0]
            elif isinstance(coord, DiscreteParameter):
                g_var = {
                    "name": f"var_{coord.name}_{idx}",
                    "type": "continuous"
                    if isinstance(coord, ContinuousParameter)
                    else "discrete",
                    "domain": tuple(coord.elements),
                }
                g_len = max(coord.elements) - min(coord.elements)
            elif isinstance(coord, FiniteParameter):
                g_var = {
                    "name": f"var_{coord.name}_{idx}",
                    "type": "continuous"
                    if isinstance(coord, ContinuousParameter)
                    else "discrete",
                    "domain": tuple(coord.elements),
                }
                g_len = max(coord.elements) - min(coord.elements)
            else:
                g_var = None
                g_len = None
            if g_var is not None:
                bounds.append(g_var)
                lengths.append(g_len)
        return bounds, lengths

    def __init__(
        self,
        parameter_model: HyperparameterMap,
        training_data: TrajectoriesData,
        n_samps=10,
        **kwargs,
    ):
        super(BayesianOptTuner, self).__init__(parameter_model, training_data, **kwargs)
        self.n_samps = n_samps

    def tune(
        self,
        nattempts=100,
        scoring_func: Callable[
            [TrajectoriesData, TrajectoriesData], float
        ] = TrajectoryScoring.end_point_score,
    ) -> TuneResults:
        import GPyOpt
        import GPy

        # get GPyOpt domain
        bounds, lengthscales = self.make_bounds(self._parameter_model.parameter_space)

        # create the sampler and send it the parameters
        sampling = self.tune_sampling(nattempts, scoring_func)
        next(sampling)

        def gpy_obj(param):
            """function for the optimizer
            TODO: deal with logspace correctly
            """
            try:
                val = sampling.send(tuple(param.flatten()))
                next(sampling)
                return val
            except StopIteration:
                return 1e10
            except Exception as exc:
                print(f"Error:<{exc}>")
                return 1e10

        kern = GPy.kern.RBF(
            len(bounds),
            variance=1.0,
            lengthscale=np.array(lengthscales) / 5.0,
            ARD=True,
        )
        self.model = GPyOpt.models.gpmodel.GPModel(kernel=kern, verbose=False)
        self.bopt = GPyOpt.methods.BayesianOptimization(
            gpy_obj, domain=bounds, model=self.model, verbose=False
        )
        self.bopt.run_optimization(max_iter=nattempts)
        return self.best_result
