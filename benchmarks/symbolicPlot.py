import matplotlib.pyplot as plt
import numpy as np
# this is the convenience function
import torch

from autokoopman import auto_koopman
# for a complete example, let's create an example dataset using an included benchmark system
from autokoopman.benchmark import bio2, fhn, lalo20, prde20, robe21, spring, pendulum, trn_constants
from benchmarks.glop import Glop
import random
import copy
import sys


def get_training_data(bench, param_dict):
    init_states = get_init_states(bench, param_dict["train_size"])
    if bench._input_vars:
        steps = []
        for low, high in zip(bench.input_set_low, bench.input_set_high):
            if bench.input_type == "step":
                params = np.random.uniform(low, high, size=(param_dict["train_size"], 3))
                steps += [make_input_step(*p, bench.teval) for p in params]
            elif bench.input_type == "rand":
                steps += [make_random_input(low, high, bench.teval) for i in range(param_dict["train_size"])]
            else:
                sys.exit("Please set an input type for your benchmark")
        training_data = bench.solve_ivps(initial_states=init_states, inputs=steps, teval=bench.teval)
    else:
        training_data = bench.solve_ivps(initial_states=init_states, tspan=[0.0, 10.0],
                                         sampling_period=param_dict["samp_period"])

    return training_data


def get_init_states(bench, size, init_seed=0):
    if hasattr(bench, 'init_constrs'):
        init_states = []
        for i in range(size):
            init_state_dict = glop_init_states(bench, i + init_seed)
            init_state = []
            for name in bench.names:
                init_state.append(init_state_dict[name])
            init_states.append(init_state)
        init_states = np.array(init_states)
    else:
        init_states = np.random.uniform(low=bench.init_set_low,
                                        high=bench.init_set_high, size=(size, len(bench.names)))

    return init_states


def glop_init_states(bench, seed):
    constrs = []
    for constr in bench.init_constrs:
        constrs.append(constr)
    for i, (name, init_low, init_high) in enumerate(zip(bench.names, bench.init_set_low, bench.init_set_high)):
        low_constr = f"{name} >= {init_low}"
        high_constr = f"{name} <= {init_high}"
        constrs.extend([low_constr, high_constr])

    glop = Glop(bench.names, constrs)
    pop_item = random.randrange(len(bench.names))
    names, init_set_low, init_set_high = copy.deepcopy(bench.names), copy.deepcopy(bench.init_set_low), copy.deepcopy(
        bench.init_set_high)
    names.pop(pop_item)
    init_set_low.pop(pop_item)
    init_set_high.pop(pop_item)
    for i, (name, init_low, init_high) in enumerate(zip(names, init_set_low, init_set_high)):
        glop.add_tend_value_obj_fn(name, [init_low, init_high], seed + i)

    glop.minimize()

    sol_dict = glop.get_all_sols()
    return sol_dict


def get_trajectories(bench, iv, samp_period):
    # get the model from the experiment results
    model = experiment_results['tuned_model']

    if bench._input_vars:
        test_inp = np.sin(np.linspace(0, 10, 200))

        # simulate using the learned model
        trajectory = model.solve_ivp(
            initial_state=iv,
            inputs=test_inp,
            teval=bench.teval,
        )

    else:
        # simulate using the learned model
        trajectory = model.solve_ivp(
            initial_state=iv,
            tspan=(0.0, 10.0),
            sampling_period=samp_period
        )

    return trajectory


def get_true_trajectories(bench, iv, samp_period):
    if bench._input_vars:
        test_inp = np.sin(np.linspace(0, 10, 200))
        # simulate the ground truth for comparison
        true_trajectory = bench.solve_ivp(
            initial_state=iv,
            inputs=test_inp,
            teval=bench.teval,
        )

    else:
        # simulate the ground truth for comparison
        true_trajectory = bench.solve_ivp(
            initial_state=iv,
            tspan=(0.0, 10.0),
            sampling_period=samp_period
        )

    return true_trajectory


def make_input_step(duty, on_amplitude, off_amplitude, teval):
    """produce a step response input signal for the pendulum"""
    length = len(teval)
    inp = np.zeros((length,))
    phase_idx = int(length * duty)
    inp[:phase_idx] = on_amplitude
    inp[phase_idx:] = off_amplitude
    return inp


def make_random_input(low, high, teval):
    length = len(teval)
    inp = np.zeros((length,))
    for i in range(len(inp)):
        inp[i] = np.random.uniform(low, high)
    return inp


def plot(trajectories):
    plt.figure(figsize=(10, 6))
    # plot the results
    for i, (label, trajectory) in enumerate(trajectories.items()):
        plt.plot(trajectory.states.T[0], trajectory.states.T[1], label=label)
        # plt.plot(trajectory.states[:, 1], label=label)

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.grid()
    plt.legend()
    plt.title("Test Trajectory Plot")
    plt.show()


if __name__ == '__main__':
    benches = [trn_constants.TRNConstants()]
    obs_types = ['id', 'poly']
    for benchmark in benches:
        iv = get_init_states(benchmark, 1, 1000)[0]
        true_trajectory = get_true_trajectories(benchmark, iv, 0.1)
        trajectories = {'true_trajectory': true_trajectory}
        for obs in obs_types:
            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)
            if obs == 'deep':
                opt = 'bopt'
            else:
                opt = 'grid'
            param_dict = {"train_size": 10, "samp_period": 0.1, "obs_type": obs, "opt": opt, "n_obs": 200,
                          "grid_param_slices": 5, "n_splits": 5, "rank": (1, 200, 40)}
            # generate training data
            training_data = get_training_data(benchmark, param_dict)
            # learn model from data
            experiment_results = auto_koopman(
                training_data,  # list of trajectories
                sampling_period=param_dict["samp_period"],  # sampling period of trajectory snapshots
                obs_type=param_dict["obs_type"],  # use Random Fourier Features Observables
                opt=param_dict["opt"],  # grid search to find best hyperparameters
                n_obs=param_dict["n_obs"],  # maximum number of observables to try
                max_opt_iter=200,  # maximum number of optimization iterations
                grid_param_slices=param_dict["grid_param_slices"],
                # for grid search, number of slices for each parameter
                n_splits=param_dict["n_splits"],  # k-folds validation for tuning, helps stabilize the scoring
                rank=param_dict["rank"]  # rank range (start, stop, step) DMD hyperparameter
            )

            trajectory = get_trajectories(benchmark, iv, 0.1)
            trajectories[f'{obs} trajectory'] = trajectory
        plot(trajectories)
