import matplotlib.pyplot as plt
from autokoopman import auto_koopman
# for a complete example, let's create an example dataset using an included benchmark system
from autokoopman.benchmark import coupledVanderPol

from numpy.linalg import norm
import statistics
import os
import csv
import time
import sys

import random
import torch
import numpy as np


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


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


def get_init_states(bench, size):
    init_states = np.random.uniform(low=bench.init_set_low,
                                    high=bench.init_set_high, size=(size, len(bench.names)))

    return init_states


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
        # simulate the ground truth for comparison
        true_trajectory = bench.solve_ivp(
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
        # simulate the ground truth for comparison
        true_trajectory = bench.solve_ivp(
            initial_state=iv,
            tspan=(0.0, 10.0),
            sampling_period=samp_period
        )

    return trajectory, true_trajectory


def test_trajectories(bench, num_tests, samp_period):
    euc_norms = []
    for j in range(num_tests):
        iv = get_init_states(bench, 1)[0]
        try:
            trajectory, true_trajectory = get_trajectories(bench, iv, samp_period)
            y_true = np.matrix.flatten(true_trajectory.states)
            y_pred = np.matrix.flatten(trajectory.states)
            euc_norm = norm(y_true - y_pred) / norm(y_true)
            euc_norms.append(euc_norm)

        except:
            print("ERROR--solve_ivp failed (likely unstable model)")
            euc_norms.append(np.infty)

    return statistics.mean(euc_norms)


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


def plot(trajectory, true_trajectory, var_1, var_2):
    plt.figure(figsize=(10, 6))
    # plot the results
    if var_2 == -1:  # plot against time
        plt.plot(trajectory.states[:, var_1], label='Trajectory Prediction')
        plt.plot(true_trajectory.states[:, var_1], label='Ground truth')
    else:
        plt.plot(trajectory.states.T[var_1], trajectory.states.T[var_2], label='Trajectory Prediction')
        plt.plot(true_trajectory.states.T[var_1], true_trajectory.states.T[var_2], label='Ground Truth')

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.grid()
    plt.legend()
    plt.title("Bio2 Test Trajectory Plot")
    plt.show()


def plot_trajectory(bench, var_1=0, var_2=-1, seed=100):
    iv = get_init_states(bench, 1)[0]
    trajectory, true_trajectory = get_trajectories(bench, iv, param_dict["samp_period"])
    plot(trajectory, true_trajectory, var_1, var_2)


if __name__ == '__main__':
    benches = [coupledVanderPol.CoupledVanderPol()]
    obs_types = ['rff', 'deep']

    for benchmark in benches:
        result = [benchmark.name, ""]
        for obs in obs_types:
            set_seed()
            if obs == 'deep':
                opt = 'bopt'
            else:
                opt = 'grid'
            param_dict = {"train_size": 10, "samp_period": 0.1, "obs_type": obs, "opt": opt, "n_obs": 100,
                          "grid_param_slices": 5, "n_splits": 5, "rank": (1, 200, 40)}
            # generate training data
            training_data = get_training_data(benchmark, param_dict)
            start = time.time()
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
            end = time.time()

            euc_norm = test_trajectories(benchmark, 10, param_dict["samp_period"])
            comp_time = round(end - start, 3)

            print(benchmark.name)
            print(f"observables type: {obs}")
            print(f"The average euc norm perc error is {round(euc_norm * 100, 2)}%")
            print("time taken: ", comp_time)
