"""
Main AutoKoopman Function (Convenience Function)
"""
from typing import Callable, Union, Sequence, Optional, Tuple

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
    TrajectoryScoring,
)
from autokoopman.estimator.koopman import KoopmanDiscEstimator
from autokoopman.tuner.gridsearch import GridSearchTuner
from autokoopman.tuner.montecarlo import MonteCarloTuner
from autokoopman.tuner.bayesianopt import BayesianOptTuner
from autokoopman.core.observables import KoopmanObservable
from autokoopman.core.format import hide_prints

__all__ = ["auto_koopman"]


# valid string identifiers for the autokoopman magic
obs_types = {"rff", "poly", "quadratic", "id", "deep"}
opt_types = {"grid", "monte-carlo", "bopt"}
scoring_func_types = {"total", "end", "relative"}


def get_scoring_func(score_name):
    """resolve scoring function from name or callable type"""
    # if callable, just return it
    if callable(score_name):
        return score_name
    if score_name == "total":
        return TrajectoryScoring.total_score
    elif score_name == "end":
        return TrajectoryScoring.end_point_score
    elif score_name == "relative":
        return TrajectoryScoring.relative_score
    else:
        raise ValueError(
            f"Scoring function name {score_name} is not in available list (names are {scoring_func_types})"
        )


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
    elif obs_type == "poly":
        return ParameterSpace(
            "koopman-polynomial",
            [
                DiscreteParameter("degree", 1, 5),
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
    elif obs_type == "poly":
        observables = kobs.PolynomialObservable(dim, hyperparams[0])
        return KoopmanDiscEstimator(
            observables, sampling_period, dim, rank=hyperparams[1]
        )
    elif obs_type == "id":
        observables = kobs.IdentityObservable()
        return KoopmanDiscEstimator(
            observables, sampling_period, dim, rank=hyperparams[0]
        )
    else:
        raise ValueError(f"unknown observables type {obs_type}")


def auto_koopman(
    training_data: Union[TrajectoriesData, Sequence[np.ndarray]],
    inputs_training_data: Optional[Sequence[np.ndarray]] = None,
    sampling_period: Optional[float] = None,
    opt: Union[str, HyperparameterTuner] = "monte-carlo",
    max_opt_iter: int = 100,
    max_epochs: int = 500,
    n_splits: Optional[int] = None,
    obs_type: Union[str, KoopmanObservable] = "rff",
    cost_func: Union[
        str, Callable[[TrajectoriesData, TrajectoriesData], float]
    ] = "total",
    n_obs: int = 100,
    rank: Optional[Union[Tuple[int, int], Tuple[int, int, int]]] = None,
    grid_param_slices: int = 10,
    lengthscale: Tuple[float, float] = (1e-4, 1e1),
    enc_dim: Tuple[int, int, int] = (2, 64, 16),
    n_layers: Tuple[int, int, int] = (1, 8, 2),
    torch_device: Optional[str] = None,
    verbose: bool = True,
):
    """
    AutoKoopman Convenience Function
        This is an interface to the dynamical systems learning functionality of the AutoKoopman library. The user can select
        estimators classes at a high level. A tuner can be chosen to find the best hyperparameter values.

    :param training_data: training trajectories data from which to learn the system
    :param inputs_training_data: optional input trajectories data from which to learn the system (this isn't needed if the training data has inputs already)
    :param sampling_period: (for discrete time system) sampling period of training data
    :param opt: hyperparameter optimizer {"grid", "monte-carlo", "bopt"}
    :param max_opt_iter: maximum iterations for the tuner to use
    :param max_epochs: maximum number of training epochs
    :param n_splits: (for optimizers) if set, switches to k-folds bootstrap validation for the hyperparameter tuning. This is useful for things like RFF tuning where the results have noise.
    :param obs_type: (for koopman) koopman observables to use {"rff", "quadratic", "poly", "id", "deep"}
    :param cost_func: cost function to use for hyperparameter optimization
    :param  n_obs: (for koopman) number of observables to use (if applicable)
    :param rank: (for koopman) rank range (start, stop) or (start, stop, step)
    :param grid_param_slices: (for grid tuner) resolution to slice continuous valued parameters into
    :param lengthscale: (for RFF observables) RFF kernel lengthscale
    :param enc_dim: (for deep learning) number of dimensions in the latent space
    :param n_layers: (for deep learning) number of hidden layers in the encoder / decoder
    :param torch_device: (for deep learning) specify torch compute device
    :param verbose: whether to print progress and messages

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

    training_data, sampling_period = _sanitize_training_data(
        training_data, inputs_training_data, sampling_period, opt, obs_type
    )

    # get the hyperparameter map
    if obs_type in {"deep"}:
        modelmap = _deep_model_map(
            training_data, max_epochs, n_obs, enc_dim, n_layers, torch_device, verbose
        )
    else:
        modelmap = _edmd_model_map(
            training_data, rank, obs_type, n_obs, lengthscale, sampling_period
        )

    # setup the tuner
    if opt == "grid":
        gt = GridSearchTuner(
            modelmap,
            training_data,
            n_samps=grid_param_slices,
            n_splits=n_splits,
            verbose=verbose,
        )
    elif opt == "monte-carlo":
        gt = MonteCarloTuner(
            modelmap, training_data, n_splits=n_splits, verbose=verbose
        )
    elif opt == "bopt":
        gt = BayesianOptTuner(modelmap, training_data, verbose=verbose)
    else:
        raise ValueError(f"could not match a tuner to the string {opt}")
    
    with hide_prints():
        res = gt.tune(nattempts=max_opt_iter, scoring_func=get_scoring_func(cost_func))

    # pack results into out custom output
    result = {
        "tuned_model": res["model"].model,
        "model_class": modelmap.parameter_space.name,
        "hyperparameters": [param.name for param in modelmap.parameter_space],
        "hyperparameter_values": res["param"],
        "tuner_score": res["score"],
        "tuner": gt,
        "estimator": res["model"],
        "has_errors": len(gt.error_messages) > 0,
        "error_messages": gt.error_messages,
    }
    return result


def _deep_model_map(
    training_data: TrajectoriesData,
    epochs,
    obs_dim,
    enc_dim,
    nlayers,
    torch_device,
    verbose,
) -> HyperparameterMap:
    import autokoopman.estimator.deepkoopman as dk

    pspace = ParameterSpace(
        "koopman-deep",
        [
            # DiscreteParameter("obs_dim", *obs_dim),
            DiscreteParameter("enc_dim", *enc_dim),
            DiscreteParameter("num_layers", *nlayers),
        ],
    )

    # system dimension
    dim = len(training_data.state_names)
    input_dim = (
        len(training_data.input_names) if training_data.input_names is not None else 0
    )

    class _ModelMap(HyperparameterMap):
        def __init__(self):
            self.names = training_data.state_names
            super(_ModelMap, self).__init__(pspace)

        def get_model(self, hyperparams: Sequence):
            return dk.DeepKoopman(
                state_dim=dim,
                input_dim=input_dim,
                hidden_dim=hyperparams[0],
                max_iter=epochs,
                lr=1e-3,
                hidden_enc_dim=64,
                num_hidden_layers=hyperparams[1],
                pred_loss_weight=1.0,
                metric_loss_weight=0.1,
                torch_device=torch_device,
                # turning this off because of review feedback
                verbose=False,
                display_progress=False,  # don't nest progress bars
            )

    # get the hyperparameter map
    return _ModelMap()


def _edmd_model_map(
    training_data, rank, obs_type, n_obs, lengthscale, sampling_period
) -> HyperparameterMap:
    """model map for eDMD based methods

    :param training_data:
    :param rank: set of ranks to try (of DMD rank parameter)
    :param obs_type:
    :param n_obs: some obs type require a number of observables
    :sampling_period:

    :returns: hyperparameter map
    """
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
    return _ModelMap()


def _sanitize_training_data(
    training_data, inputs_training_data, sampling_period, opt, obs_type
):
    """auto_koopman input sanitization"""

    # if sampling period is None AND discrete system is wanted
    if sampling_period is None:
        sampling_period = np.infty
        for t in training_data:
            sampling_period = min(sampling_period, min(np.diff(t.times)))

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

    return training_data, sampling_period
