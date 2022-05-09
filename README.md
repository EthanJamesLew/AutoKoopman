# AutoKoopman

![Conda CI Workflow](https://github.com/EthanJamesLew/AutoKoopman/actions/workflows/python-package-conda.yml/badge.svg)

## Overview
AutoKoopman is a python library for the use of Koopman operator methods for data-driven dynamical systems analysis and control. The library
has convenient functions to learn systems using a few lines of code. It has a variety of linearization methods under
shared class interfaces. These methods are pluggable into hyperparameter optimizers which can automate the process of model
optimization.

## Use Cases
A systems engineer / researcher who wishes to leverage data-driven dynamical systems techniques. The user may
have measurements of their system with no prior model.
* System Prediction - the user can simulate a model learned from their measurements. They use popular techniques like DMD and SINDy out of the box, and implement their own methods to plug into the provided analysis infrastructure (e.g. hyperparameter optimization, visualization).
* System Linearization - the user can get a linear representation of their system in its original states or koopman observables. They can use this linear form to perform tasks like controller synthesis and system reachability.

## Installation

The module requires python 3.8 or higher. With pip installed, run
```shell
pip install .
```
at the repo root. Run
```shell
python -c "import autokoopman"
```
to ensure that the module can be imported.

## Examples

### Complete Example
AutoKoopman has a convenience function `auto_koopman` that can learn dynamical systems from data in one call, given
training data of trajectories (list of arrays),
```python
import matplotlib.pyplot as plt
import numpy as np

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
```


## Documentation

[AutoKoopman Documentation](https://ethanjameslew.github.io/AutoKoopman/)
