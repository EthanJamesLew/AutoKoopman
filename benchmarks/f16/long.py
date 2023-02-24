# the notebook imports
import os

import matplotlib.pyplot as plt
import numpy as np

# this is the convenience function
from autokoopman import auto_koopman

import autokoopman.benchmark.fhn as fhn
import autokoopman.core.trajectory as traj
from sklearn.metrics import mean_squared_error
import statistics

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'data/trainingdata/long/trainingDataUniform20.csv')
training_data = traj.TrajectoriesData.from_csv(filename)

# learn model from data
experiment_results = auto_koopman(
    training_data,          # list of trajectories
    sampling_period=0.1,    # sampling period of trajectory snapshots
    obs_type="id",         # use Random Fourier Features Observables
    opt="grid",             # grid search to find best hyperparameters
    n_obs=200,              # maximum number of observables to try
    max_opt_iter=200,       # maximum number of optimization iterations
    grid_param_slices=5,   # for grid search, number of slices for each parameter
    n_splits=5,             # k-folds validation for tuning, helps stabilize the scoring
    rank=(1, 200, 40)       # rank range (start, stop, step) DMD hyperparameter
)

# get the model from the experiment results
model = experiment_results['tuned_model']
# simulate using the learned model

filename = os.path.join(dirname, 'data/testdata/long.csv')
true_trajectories = traj.TrajectoriesData.from_csv(filename)

for i in range(0, 9):
    init_s = true_trajectories[i].states[0]

    iv = init_s
    trajectory = model.solve_ivp(
        initial_state=iv,
        tspan=(0.0, 15),
        sampling_period=0.1
    )


    print(len(true_trajectories[i].states))
    print(len(trajectory.states))