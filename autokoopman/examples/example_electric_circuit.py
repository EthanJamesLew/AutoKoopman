import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path
import sys
sys.path.insert(0, '/home/niklas/Documents/Repositories/AutoKoopman')
from autokoopman import auto_koopman
from autokoopman.core import trajectory

"""For this script to run the measurement data needs to be downloaded from 
https://drive.google.com/drive/folders/1blkHkK3tMG2lKaq3POWtxquYHYw6kf6i?usp=sharing 
and the variable path below needs to be updated accordingly"""

# load data measured from a F1tenth racecar
path = '/home/niklas/Documents/Work/Projekte/KoopmanModels/ElectricCircuit'
cnt = 1
data = []
ids = []

while True:
    dirpath = os.path.join(path, 'measurement_' + str(cnt))
    if os.path.isdir(dirpath):
        traj = np.asarray(pd.read_csv(os.path.join(dirpath, 'trajectory.csv')))
        input = np.asarray(pd.read_csv(os.path.join(dirpath, 'input.csv')))
        time = np.asarray(pd.read_csv(os.path.join(dirpath, 'time.csv')))
        dt = time[1][0] - time[0][0]
        data.append(trajectory.UniformTimeTrajectory(traj, dt, ['nin2', 'neg', 'output']))
        ids.append(str(cnt))
        cnt += 1
    else:
        break

training_data = trajectory.UniformTimeTrajectoriesData(dict(zip(ids[0:1], data[0:1])))

# learn model from data
experiment_results = auto_koopman(
    training_data,          # list of trajectories
    sampling_period=dt,     # sampling period of trajectory snapshots
    obs_type="rff",         # use Random Fourier Features Observables
    opt="grid",             # grid search to find best hyperparameters
    n_obs=100,              # maximum number of observables to try
    max_opt_iter=20,        # maximum number of optimization iterations
    grid_param_slices=5,    # for grid search, number of slices for each parameter
    rank=(1, 10, 3)         # rank range (start, stop, step) DMD hyperparameter
)

# get the model from the experiment results
model = experiment_results['tuned_model']

# simulate using the learned model
traj = training_data._trajs['1'].states
iv = traj[0]
trajectory = model.solve_ivp(
    initial_state=iv,
    tspan=(0.0, dt * traj.shape[0]),
    sampling_period=dt
)

# plot the results
plt.plot(traj[:, 2])
plt.plot(trajectory.states[:, 2])
plt.show()