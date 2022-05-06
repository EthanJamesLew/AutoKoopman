"""
Main AutoKoopman Function (Convenience Function)
"""
from typing import Union, Sequence, Any

import numpy as np

import autokoopman.core.observables as kobs
from autokoopman import TrajectoriesData, UniformTimeTrajectory
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


obs_types = {"rff", "quadratic", "id"}
tuner_types = {"grid", "monte-carlo"}


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
    sampling_period=0.05,
    opt: Union[str, HyperparameterTuner] = "monte-carlo",
    max_opt_iter=100,
    n_splits=None,
    obs_type: Union[str, Any] = "rff",
    n_obs=100,
    rank=None,
    grid_param_slices=10,
    lengthscale=(1e-4, 1e1),
):
    """
    AutoKoopman Convenience Function

    This is an interface to the dynamical systems learning functionality of the AutoKoopman library. The user can select
    estimators classes at a high level. A tuner can be chosen to find the best hyperparameter values.

    :param training_data: training trajectories data from which to learn the system
    :param sampling_period: (for discrete time system) sampling period of training data
    :param opt: hyperparameter optimizer
    :param max_opt_iter: maximum iterations for the tuner to use
    :param n_splits: (for optimizers) if set, switches to k-folds bootstrap validation for the hyperparameter tuning. This
    is useful for things like RFF tuning where the results have noise.
    :param obs_type: (for koopman) koopman observables to use
    :param  n_obs: (for koopman) number of observables to use (if applicable)
    :param rank: (for koopman) rank range (start, stop) or (start, stop, step)
    :param grid_param_slices: (for grid tuner) resolution to slice continuous valued parameters into
    :param lengthscale: (for RFF observables) RFF kernel lengthscale
    """

    # sanitize the input
    # check the strings
    if isinstance(obs_type, str):
        assert (
            obs_type in obs_types
        ), f"observable name {obs_type} is unknown (valid ones are {obs_types})"
    if isinstance(opt, str):
        assert (
            opt in tuner_types
        ), f"tuner name {opt} is unknown (valid ones are {tuner_types})"

    # convert the data to autokoopman trajectories
    if isinstance(training_data, TrajectoriesData):
        pass
    elif isinstance(training_data, dict):
        training_data = TrajectoriesData(
            {
                k: UniformTimeTrajectory(v, sampling_period=sampling_period)
                for k, v in training_data.items()
            }
        )
    else:
        training_data = TrajectoriesData(
            {
                idx: UniformTimeTrajectory(di, sampling_period=sampling_period)
                for idx, di in enumerate(training_data)
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
