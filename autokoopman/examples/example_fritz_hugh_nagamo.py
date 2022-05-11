import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '/home/niklas/Documents/Repositories/AutoKoopman')

# this is the convenience function
from autokoopman import auto_koopman

# for a complete example, let's create an example dataset using an included benchmark system
import autokoopman.benchmark.fhn as fhn
fhn = fhn.FitzHughNagumo()
training_data = fhn.solve_ivps(
    initial_states=np.random.uniform(low=-2.0, high=2.0, size=(10, 2)),
    tspan=[0.0, 10.0],
    sampling_period=0.1
)

# learn model from data
experiment_results = auto_koopman(
    training_data,          # list of trajectories
    sampling_period=0.1,    # sampling period of trajectory snapshots
    obs_type="rff",         # use Random Fourier Features Observables
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
iv = [0.5, 0.1]
trajectory = model.solve_ivp(
    initial_state=iv,
    tspan=(0.0, 10.0),
    sampling_period=0.1
)

# simulate the ground truth for comparison
true_trajectory = fhn.solve_ivp(
    initial_state=iv,
    tspan=(0.0, 10.0),
    sampling_period=0.1
)

# plot the results
plt.plot(*trajectory.states.T)
plt.plot(*true_trajectory.states.T)
plt.show()
