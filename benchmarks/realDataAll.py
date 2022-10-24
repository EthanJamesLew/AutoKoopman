import pickle
import numpy as np
import os.path
import sys
import random
import time
import csv
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import statistics

from autokoopman import auto_koopman
import autokoopman.core.trajectory as traj

"""For this script to run the measurement data needs to be downloaded from 
https://drive.google.com/drive/folders/1blkHkK3tMG2lKaq3POWtxquYHYw6kf6i?usp=sharing 
and the variable PATH below needs to be updated accordingly"""

PATH = '/Users/b6062805/Documents/Koopman/RealDataModels'


def load_data(benchmark):
    """load the measured data"""

    path = os.path.join(PATH, benchmark)
    cnt = 1
    data = []

    while True:
        dirpath = os.path.join(path, 'measurement_' + str(cnt))
        if os.path.isdir(dirpath):
            states = np.asarray(pd.read_csv(os.path.join(dirpath, 'trajectory.csv')))
            inputs = np.asarray(pd.read_csv(os.path.join(dirpath, 'input.csv')))
            time = np.asarray(pd.read_csv(os.path.join(dirpath, 'time.csv')))
            time = np.resize(time, (time.shape[0],))
            data.append(traj.Trajectory(time[:-1], states[:-1, :], inputs))
            cnt += 1
        else:
            break

    if len(data) > 100:
        data = data[0:100]

    return data


def split_data(data, num_test=10):
    """randomly split data into training and test set"""

    np.random.seed(0)
    ind = random.sample(range(0, len(data)), num_test)

    test_data = [data[i] for i in ind]
    training_data = [data[i] for i in range(0, len(data)) if i not in ind]

    ids = np.arange(0, len(training_data)).tolist()
    training_data = traj.TrajectoriesData(dict(zip(ids, training_data)))

    ids = np.arange(0, len(test_data)).tolist()
    test_data = traj.TrajectoriesData(dict(zip(ids, test_data)))

    print(len(training_data))
    return training_data, test_data


def train_model(data, obs_type):
    """train the Koopman model using the AutoKoopman library"""

    # learn model from data
    experiment_results = auto_koopman(
        data,  # list of trajectories
        obs_type=obs_type,
        n_splits=5,
        opt='grid',
        n_obs=200,
        rank=(1, 200, 40),
        grid_param_slices=5,
        max_opt_iter=200
    )

    # get the model from the experiment results
    model = experiment_results['tuned_model']

    return model


def compute_error(model, test_data):
    """compute error between model prediction and real data"""

    mses = []
    perc_errors = []

    # loop over all test trajectories
    tmp = list(test_data._trajs.values())

    for t in tmp:
        # simulate using the learned model
        iv = t.states[0, :]
        start_time = t.times[0]
        end_time = t.times[len(t.times) - 1]
        teval = np.linspace(start_time, end_time, len(t.times))

        trajectory = model.solve_ivp(
            initial_state=iv,
            tspan=(start_time, end_time),
            sampling_period=t.times[1] - t.times[0],
            inputs=t.inputs,
            teval=teval
        )

        # compute error
        mse = mean_squared_error(t.states, trajectory.states)
        mses.append(mse)
        perc_error = mean_absolute_percentage_error(t.states, trajectory.states)
        perc_errors.append(perc_error)

    # take mean over all errors
    perc_error = statistics.mean(perc_errors)
    mse = statistics.mean(mses)

    return perc_error, mse


def store_data(row, filename='real_data'):
    with open(f'data/{filename}', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def store_data_heads(row, filename='real_data'):
    if not os.path.exists('../notebooks/data'):
        os.makedirs('../notebooks/data')
    with open(f'data/{filename}', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(row)


if __name__ == '__main__':

    # initialization
    # benchmarks = ['ElectricCircuit', 'F1tenthCar', 'Robot']
    benchmarks = ['ElectricCircuit']
    obs_types = ['id', "poly", "rff"]
    store_data_heads(["", ""] + ["perc_error", "time(s)", ""] * 4)

    # loop over all benchmarks
    for i in range(1):
        store_data([f"Iteration {i + 1}"])
        for benchmark in benchmarks:

            print(' ')
            print(benchmark, ' --------------------------------------------------------------')
            print(' ')

            # load data
            data = load_data(benchmark)

            # split into training and validation set
            n_test = min(10, np.floor(0.4 * len(data)).astype(int))
            training_data, test_data = split_data(data, n_test)

            # loop over the different observable types
            result = [benchmark, ""]

            for obs in obs_types:
                np.random.seed(0)
                # train the Koopman model
                start = time.time()
                model = train_model(training_data, obs)
                end = time.time()

                comp_time = round(end - start, 3)

                # compute error
                perc_error, mse = compute_error(model, test_data)

                # store and print results
                result.append(perc_error)
                result.append(comp_time)
                result.append("")

                print(obs, ": ", perc_error, " (error), ", comp_time, " (time)")

            store_data(result)
