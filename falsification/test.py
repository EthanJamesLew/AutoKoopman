# the notebook imports
import csv
import os
import time
import random

import torch
import matplotlib.pyplot as plt
import numpy as np

# this is the convenience function
from autokoopman import auto_koopman

import autokoopman.benchmark.fhn as fhn
import autokoopman.core.trajectory as traj
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from numpy.linalg import norm
import statistics


def get_train_data(filepath):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, filepath)
    training_data = traj.TrajectoriesData.from_csv(filename)

    return training_data, dirname

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


if __name__ == '__main__':

    set_seed()

    training_data, dirname = get_train_data('/Users/b6062805/Documents/Koopman/AutoKoopman/falsification/data/AT1_trajectories.csv')

    if training_data.n_trajs > 3:
        n_splits = int(training_data.n_trajs / 2) if training_data.n_trajs % 2 == 0 else None
    else:
        n_splits = None

    experiment_results = auto_koopman(
        training_data,  # list of trajectories
        sampling_period=1,  # sampling period of trajectory snapshots
        obs_type="rff",  # use Random Fourier Features Observables
        opt="grid",  # grid search to find best hyperparameters
        n_obs=20,  # maximum number of observables to try
        max_opt_iter=200,  # maximum number of optimization iterations
        grid_param_slices=5,
        # for grid search, number of slices for each parameter
        rank=(1,20,4),  # rank range (start, stop, step) DMD hyperparameter
        n_splits=n_splits,
        verbose=False,
    )

    model = experiment_results['tuned_model']
    # get evolution matrices
    A, B = model.A, model.B
    w = model.obs_func.observables[1].w
    u = model.obs_func.observables[1].u

    params = experiment_results['hyperparameters']
    paramVals = experiment_results['hyperparameter_values']
    score = experiment_results['tuner_score']

    print(score)
    print(params)
    print(paramVals)
