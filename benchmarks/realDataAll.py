import pickle
import numpy as np
import os.path
import sys
import random
import time
import csv
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from numpy.linalg import norm
import statistics

from autokoopman import auto_koopman
import autokoopman.core.trajectory as traj

"""For this script to run the measurement data needs to be downloaded from 
https://drive.google.com/drive/folders/1blkHkK3tMG2lKaq3POWtxquYHYw6kf6i?usp=sharing 
and the variable PATH below needs to be updated accordingly"""

PATH = '/Users/b6062805/realData'


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

    random.seed(0)
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

    dt = data._trajs[0].times[1] - data._trajs[0].times[0]

    if obs_type == 'deep':
        opt = 'bopt'
    else:
        opt = 'grid'

    # learn model from data
    experiment_results = auto_koopman(
        data,  # list of trajectories
        sampling_period=dt,
        obs_type=obs_type,
        opt=opt,
        n_obs=200,
        rank=(1, 20, 1),
        grid_param_slices=5,
        max_opt_iter=200
    )

    # get the model from the experiment results
    model = experiment_results['tuned_model']

    return model


def compute_error(model, test_data):
    """compute error between model prediction and real data"""
    euc_norms = []

    # loop over all test trajectories
    tmp = list(test_data._trajs.values())

    for t in tmp:
        try:
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
            y_true = np.matrix.flatten(t.states)
            y_pred = np.matrix.flatten(trajectory.states)
            euc_norm = norm(y_true - y_pred) / norm(y_true)
            euc_norms.append(euc_norm)
        except:
            print("ERROR--solve_ivp failed (likely unstable model)")
            # NOTE: Robot has constant 0 states, resulting in high error numbers (MSE is good)
            euc_norms.append(np.infty)

    return statistics.mean(euc_norms)


def store_data(row, filename='real_data'):
    with open(f'data/{filename}', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def store_data_heads(row, filename='real_data'):
    if not os.path.exists('data'):
        os.makedirs('data')
    with open(f'data/{filename}', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(row)


if __name__ == '__main__':

    # initialization
    benchmarks = ['ElectricCircuit', 'F1tenthCar', 'Robot']
    obs_types = ['id', 'poly', 'rff', 'deep']
    store_data_heads(["", ""] + ["euc_norm", "time(s)", ""] * len(obs_types))

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
            random.seed(0)
            torch.manual_seed(0)

            start = time.time()
            model = train_model(training_data, obs)
            end = time.time()

            # compute error
            euc_norm = compute_error(model, test_data)
            comp_time = round(end - start, 3)

            print(benchmark)
            print(f"observables type: {obs}")
            print(f"The average euc norm perc error is {round(euc_norm * 100, 2)}%")
            print("time taken: ", comp_time)
            # store and print results
            result.append(round(euc_norm * 100, 2))
            result.append(comp_time)
            result.append("")

        store_data(result)
