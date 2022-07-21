"""
Main AutoKoopman Function (Convenience Function)
"""
from typing import Union, Sequence, Optional, Tuple

import numpy as np

import autokoopman.core.observables as kobs
from autokoopman.core.trajectory import (
    TrajectoriesData,
    UniformTimeTrajectoriesData,
    UniformTimeTrajectory,
)
from autokoopman.core.tuner import (
    HyperparameterTuner,
    HyperparameterMap,
    ParameterSpace,
    ContinuousParameter,
    DiscreteParameter,
)
from autokoopman.estimator.koopman import KoopmanDiscEstimator
from autokoopman.tuner.gridsearch import GridSearchTuner
from autokoopman.tuner.montecarlo import MonteCarloTuner
from autokoopman.core.observables import KoopmanObservable

__all__ = ["auto_koopman"]


# valid string identifiers for the autokoopman magic
obs_types = {"rff", "quadratic", "id"}
opt_types = {"grid", "monte-carlo"}


def get_parameter_space(obs_type, threshold_range, rank):
    """from the myriad of user suppled switches, select the right hyperparameter space"""
    if obs_type == "rff":
        return ParameterSpace(
            "koopman-rff",
            [
                ContinuousParameter(
                    "gamma", *threshold_range, distribution="loguniform"
                ),
                DiscreteParameter("rank", *rank),
            ],
        )
    elif obs_type == "quadratic":
        return ParameterSpace(
            "koopman-quadratic",
            [
                DiscreteParameter("rank", *rank),
            ],
        )
    elif obs_type == "id":
        return ParameterSpace(
            "koopman-id",
            [
                DiscreteParameter("rank", *rank),
            ],
        )


def get_estimator(obs_type, sampling_period, dim, obs, hyperparams):
    """from the myriad of user suppled switches, select the right estimator"""
    if obs_type == "rff":
        observables = kobs.IdentityObservable() | kobs.RFFObservable(
            dim, obs, hyperparams[0]
        )
        return KoopmanDiscEstimator(
            observables, sampling_period, dim, rank=hyperparams[1]
        )
    elif obs_type == "quadratic":
        observables = kobs.IdentityObservable() | kobs.QuadraticObservable(dim)
        return KoopmanDiscEstimator(
            observables, sampling_period, dim, rank=hyperparams[0]
        )
    elif obs_type == "id":
        observables = kobs.IdentityObservable()
        return KoopmanDiscEstimator(
            observables, sampling_period, dim, rank=hyperparams[0]
        )


def auto_koopman(
    training_data: Union[TrajectoriesData, Sequence[np.ndarray]],
    inputs_training_data: Optional[Sequence[np.ndarray]] = None,
    sampling_period: float = 0.05,
    opt: Union[str, HyperparameterTuner] = "monte-carlo",
    max_opt_iter: int = 100,
    n_splits: Optional[int] = None,
    obs_type: Union[str, KoopmanObservable] = "rff",
    n_obs: int = 100,
    rank: Optional[Union[Tuple[int, int], Tuple[int, int, int]]] = None,
    grid_param_slices: int = 10,
    lengthscale: Tuple[float, float] = (1e-4, 1e1),
):
    """
    AutoKoopman Convenience Function
        This is an interface to the dynamical systems learning functionality of the AutoKoopman library. The user can select
        estimators classes at a high level. A tuner can be chosen to find the best hyperparameter values.

    :param training_data: training trajectories data from which to learn the system
    :param inputs_training_data: optional input trajectories data from which to learn the system (this isn't needed if the training data has inputs already)
    :param sampling_period: (for discrete time system) sampling period of training data
    :param opt: hyperparameter optimizer {"grid", "monte-carlo"}
    :param max_opt_iter: maximum iterations for the tuner to use
    :param n_splits: (for optimizers) if set, switches to k-folds bootstrap validation for the hyperparameter tuning. This is useful for things like RFF tuning where the results have noise.
    :param obs_type: (for koopman) koopman observables to use {"rff", "quadratic", "id"}
    :param  n_obs: (for koopman) number of observables to use (if applicable)
    :param rank: (for koopman) rank range (start, stop) or (start, stop, step)
    :param grid_param_slices: (for grid tuner) resolution to slice continuous valued parameters into
    :param lengthscale: (for RFF observables) RFF kernel lengthscale

    :returns: Tuned Model and Metadata

    Example:
        .. code-block:: python

            from autokoopman.benchmark.fhn import FitzHughNagumo
            from autokoopman import auto_koopman

            # let's build an example dataset
            data = fhn.solve_ivps(
                initial_states=[[0.0, -4.0], [1.0, 3.4], [1.0, 1.0], [0.1, -0.1]],
                tspan=[0.0, 1.0], sampling_period=0.01
            )

            # learn a system
            results = auto_koopman(
                data,
                obs_type="rff",
                opt="grid",
                n_obs=200,
                max_opt_iter=200,
                grid_param_slices=10,
                n_splits=3,
                rank=(1, 200, 20)
            )

            # results = {'tuned_model': <StepDiscreteSystem Dimensions: 2 States: [X1, X2]>,
            # 'model_class': 'koopman-rff',
            # 'hyperparameters': ['gamma', 'rank'],
            # 'hyperparameter_values': (0.004641588833612782, 21),
            # 'tuner_score': 0.14723275426562,
            # 'tuner': <autokoopman.tuner.gridsearch.GridSearchTuner at 0x7f0f92f95580>,
            # 'estimator': <autokoopman.estimator.koopman.KoopmanDiscEstimator at 0x7f0f92ff0610>}
    """

    # sanitize the input
    # check the strings
    if isinstance(obs_type, str):
        assert (
            obs_type in obs_types
        ), f"observable name {obs_type} is unknown (valid ones are {obs_types})"
    if isinstance(opt, str):
        assert (
            opt in opt_types
        ), f"tuner name {opt} is unknown (valid ones are {opt_types})"

    # convert the data to autokoopman trajectories
    if isinstance(training_data, TrajectoriesData):
        if not isinstance(training_data, UniformTimeTrajectoriesData):
            training_data = training_data.interp_uniform_time(sampling_period)
    else:
        # figure out how to add inputs
        training_iter = (
            training_data.items() if isinstance(training_data, dict) else training_data
        )
        if inputs_training_data is not None:
            training_iter = [(n, x, inputs_training_data[n]) for n, x in training_iter]
        else:
            training_iter = [(n, x, None) for n, x in training_iter]
        if isinstance(training_data, dict):
            training_data = UniformTimeTrajectoriesData(
                {
                    k: UniformTimeTrajectory(v, u, sampling_period=sampling_period)
                    for k, v, u in training_iter
                }
            )
        else:
            training_data = UniformTimeTrajectoriesData(
                {
                    idx: UniformTimeTrajectory(di, u, sampling_period=sampling_period)
                    for idx, di, u in training_iter
                }
            )

    # system dimension
    dim = len(training_data.state_names)

    # infer rank range and step
    if rank is not None:
        if len(rank) == 2:
            rank = (*rank, 1)
        else:
            rank = rank
    else:
        rank = (2, n_obs + dim, 10)

    # get the hyperparameter space
    pspace = get_parameter_space(obs_type, lengthscale, rank)

    class _ModelMap(HyperparameterMap):
        """SINDy with hyperparameters for polynomial library"""

        def __init__(self):
            self.names = training_data.state_names
            super(_ModelMap, self).__init__(pspace)

        def get_model(self, hyperparams: Sequence):
            return get_estimator(obs_type, sampling_period, dim, n_obs, hyperparams)

    # get the hyperparameter map
    modelmap = _ModelMap()

    # setup the tuner
    if opt == "grid":
        gt = GridSearchTuner(
            modelmap, training_data, n_samps=grid_param_slices, n_splits=n_splits
        )
    elif opt == "monte-carlo":
        gt = MonteCarloTuner(modelmap, training_data, n_splits=n_splits)
    else:
        raise ValueError(f"could not match a tuner to the string {opt}")

    res = gt.tune(nattempts=max_opt_iter)

    # pack results into out custom output
    result = {
        "tuned_model": res["model"].model,
        "model_class": modelmap.parameter_space.name,
        "hyperparameters": [param.name for param in modelmap.parameter_space],
        "hyperparameter_values": res["param"],
        "tuner_score": res["score"],
        "tuner": gt,
        "estimator": res["model"],
    }
    return result
