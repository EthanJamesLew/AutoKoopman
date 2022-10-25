# the notebook imports
import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np

# this is the convenience function
from autokoopman import auto_koopman

import autokoopman.benchmark.fhn as fhn
import autokoopman.core.trajectory as traj
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import statistics


def get_train_data(filepath):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, filepath)
    training_data = traj.TrajectoriesData.from_csv(filename)

    return training_data, dirname


def get_true_trajectories(filepath):
    # get the model from the experiment results
    model = experiment_results['tuned_model']
    # simulate using the learned model
    filename = os.path.join(dirname, filepath)
    true_trajectories = traj.TrajectoriesData.from_csv(filename)

    return true_trajectories, model


def test_trajectories(true_trajectories, model, tspan):
    mses = []
    perc_errors = []
    for i in range(9):
        init_s = true_trajectories[i].states[0]

        iv = init_s
        trajectory = model.solve_ivp(
            initial_state=iv,
            tspan=tspan,
            sampling_period=0.1
        )
        mse = mean_squared_error(true_trajectories[i].states, trajectory.states)
        mses.append(mse)
        perc_error = mean_absolute_percentage_error(true_trajectories[i].states, trajectory.states)
        perc_errors.append(perc_error)

    return statistics.mean(mses), statistics.mean(perc_errors)


def store_data(row, filename='black_box_data'):
    with open(f'data/{filename}', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def store_data_heads(row, filename='black_box_data'):
    if not os.path.exists('data'):
        os.makedirs('data')

    with open(f'data/{filename}', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(row)


if __name__ == '__main__':

    benches = ["Engine Control", "Longitudunal", "Ground_collision"]
    train_datas = ['f16/data/trainingdata/engine/trainingDataUniform20.csv',
                   'f16/data/trainingdata/long/trainingDataUniform20.csv',
                   'f16/data/trainingdata/gsac/trainingDataUniform400.csv']
    tspans = [(0.0, 60), (0.0, 15), (0.0, 15)]
    trajectories_filepaths = ['f16/data/testdata/checkEngine.csv', 'f16/data/testdata/long.csv',
                              'f16/data/testdata/gsac.csv']
    # obs_types = ['id', 'poly', 'rff', 'deep']
    obs_types = ['id']
    store_data_heads(["", ""] + ["perc_error", "mse", "time(s)", ""] * 4)
    for i in range(1):
        store_data([f"Iteration {i + 1}"])
        for benchmark, train_data, tspan, trajectories_filepath in zip(benches, train_datas, tspans,
                                                                       trajectories_filepaths):
            result = [benchmark, ""]
            for obs in obs_types:
                np.random.seed(0)
                training_data, dirname = get_train_data(train_data)
                start = time.time()
                # learn model from data
                experiment_results = auto_koopman(
                    training_data,  # list of trajectories
                    sampling_period=0.1,  # sampling period of trajectory snapshots
                    obs_type=obs,  # use Random Fourier Features Observables
                    opt="grid",  # grid search to find best hyperparameters
                    n_obs=200,  # maximum number of observables to try
                    max_opt_iter=200,  # maximum number of optimization iterations
                    grid_param_slices=5,  # for grid search, number of slices for each parameter
                    n_splits=5,  # k-folds validation for tuning, helps stabilize the scoring
                    rank=(1, 200, 40)  # rank range (start, stop, step) DMD hyperparameter
                )
                end = time.time()
                true_trajectories, model = get_true_trajectories(trajectories_filepath)
                mse, perc_error = test_trajectories(true_trajectories, model, tspan)

                comp_time = round(end - start, 3)
                print("time taken: ", comp_time)
                print(f"The average percentage error is {perc_error}%")

                result.append(perc_error)
                result.append(mse)
                result.append(comp_time)
                result.append("")

            store_data(result)
