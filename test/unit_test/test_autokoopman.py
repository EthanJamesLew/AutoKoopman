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


auto_config = tuple(itertools.product(obs_types, opt_types, scoring_func_types))


@pytest.mark.parametrize(
    "obs, opt, cost",
    auto_config,
)
def test_autokoopman(obs, opt, cost):
    # for a complete example, let's create an example dataset using an included benchmark system
    import autokoopman.benchmark.fhn as fhn

    fhn = fhn.FitzHughNagumo()
    np.random.seed(0)

    # given issue #29, let's make these differently sized
    ivs = np.random.uniform(low=-2.0, high=2.0, size=(20, 2))
    training_data = UniformTimeTrajectoriesData(
        {
            idx: fhn.solve_ivp(
                initial_state=iv,
                tspan=[0.0, np.random.random() + 1.0],
                sampling_period=0.1,
            )
            for idx, iv in enumerate(ivs)
        }
    )

    # learn model from data
    # make the run as short as possible but still be meaningful for catching errors
    experiment_results = auto_koopman(
        training_data,
        sampling_period=0.1,
        obs_type=obs,
        opt=opt,
        cost_func=cost,
        n_obs=20,
        max_opt_iter=2,
        grid_param_slices=2,
        n_splits=2,
        rank=(10, 12, 1),
        max_epochs=1,
        torch_device="cpu",
    )
