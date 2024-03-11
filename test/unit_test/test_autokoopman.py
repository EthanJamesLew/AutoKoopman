import itertools
from autokoopman.autokoopman import (
    obs_types,
    opt_types,
    scoring_func_types,
    auto_koopman,
)
from autokoopman.core.trajectory import UniformTimeTrajectoriesData
import numpy as np
import pytest


auto_config = tuple(
    itertools.product(obs_types, opt_types, scoring_func_types, {True, False})
)


@pytest.mark.parametrize(
    "obs, opt, cost, normalize",
    auto_config,
)
def test_autokoopman(obs, opt, cost, normalize):
    # for a complete example, let's create an example dataset using an included benchmark system
    import autokoopman.benchmark.fhn as fhn

    fhn = fhn.FitzHughNagumo()
    np.random.seed(0)

    # given issue #29, let's make these differently sized
    sp = 0.1
    ivs = np.random.uniform(low=-2.0, high=2.0, size=(20, 2))
    t_ends = [np.random.random() + 1.0 for idx in range(len(ivs))]
    lengths = [int(t_end // sp) for t_end in t_ends]
    training_data = UniformTimeTrajectoriesData(
        {
            idx: fhn.solve_ivp(
                initial_state=iv,
                tspan=[0.0, t_ends[idx]],
                sampling_period=sp,
            )
            for idx, iv in enumerate(ivs)
        }
    )

    # produce scoring weights if the cost is weighted
    if cost == "weighted":
        scoring_weights = {
            idx: np.ones((length,)) * 0.01 for idx, length in enumerate(lengths)
        }
    else:
        scoring_weights = None

    # learn model from data
    # make the run as short as possible but still be meaningful for catching errors
    experiment_results = auto_koopman(
        training_data,
        sampling_period=0.1,
        obs_type=obs,
        opt=opt,
        cost_func=cost,
        scoring_weights=scoring_weights,
        n_obs=20,
        max_opt_iter=2,
        grid_param_slices=2,
        n_splits=2,
        rank=(10, 12, 1),
        max_epochs=1,
        torch_device="cpu",
        normalize=normalize,
    )


@pytest.mark.parametrize(
    "obs, opt, cost, normalize",
    auto_config,
)
def test_autokoopman_np(obs, opt, cost, normalize):
    """test the case of autokoopman using numpy arrays"""

    # for a complete example, let's create an example dataset using an included benchmark system
    import autokoopman.benchmark.fhn as fhn

    fhn = fhn.FitzHughNagumo()
    np.random.seed(0)

    # given issue #29, let's make these differently sized
    sp = 0.1
    ivs = np.random.uniform(low=-2.0, high=2.0, size=(20, 2))
    t_ends = [np.random.random() + 1.0 for idx in range(len(ivs))]
    lengths = [int(t_end // sp) for t_end in t_ends]
    training_data = UniformTimeTrajectoriesData(
        {
            idx: fhn.solve_ivp(
                initial_state=iv,
                tspan=[0.0, t_ends[idx]],
                sampling_period=sp,
            )
            for idx, iv in enumerate(ivs)
        }
    )

    # produce scoring weights if the cost is weighted
    if cost == "weighted":
        scoring_weights = {
            idx: np.ones((length,)) * 0.01 for idx, length in enumerate(lengths)
        }
    else:
        scoring_weights = None

    # put into numpy arrays for testing
    training_data_np = []
    scoring_weights_np = []
    for name in training_data.traj_names:
        training_data_np.append(training_data[name].states)
        if scoring_weights is not None:
            scoring_weights_np.append(scoring_weights[name])

    # learn model from data
    # make the run as short as possible but still be meaningful for catching errors
    experiment_results = auto_koopman(
        training_data_np,
        sampling_period=0.1,
        obs_type=obs,
        opt=opt,
        cost_func=cost,
        scoring_weights=scoring_weights_np,
        n_obs=20,
        max_opt_iter=2,
        grid_param_slices=2,
        n_splits=2,
        rank=(10, 12, 1),
        max_epochs=1,
        torch_device="cpu",
        normalize=normalize,
    )
