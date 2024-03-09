## A Simple Linear System

```python
import sys
sys.path.append("../..")

from autokoopman import SymbolicContinuousSystem, auto_koopman
import sympy as sp
import numpy as np

class SimpleLinear(SymbolicContinuousSystem):
    def __init__(self):
        x1, x2 = sp.symbols("x1 x2")
        xdot = [
            1.2 * x1 + 0.5 * x2,
            -0.7 * x1 + 0.1 * x2
        ]
        super().__init__((x1, x2), xdot)
```

```python
sys = SimpleLinear()
training_data = sys.solve_ivps(
    initial_states=np.random.uniform(low=-2.0, high=2.0, size=(10, 2)),
    tspan=[0.0, 10.0],
    sampling_period=0.1
)
```

```python
# learn model from data
experiment_results = auto_koopman(
    training_data,          # list of trajectories
    sampling_period=0.1,    # sampling period of trajectory snapshots
    learn_continuous=True,    
    obs_type="id",         # use Random Fourier Features Observables
    opt="grid",             # grid search to find best hyperparameters
    n_obs=200,              # maximum number of observables to try
    max_opt_iter=200,       # maximum number of optimization iterations
    grid_param_slices=5,   # for grid search, number of slices for each parameter
    n_splits=5,             # k-folds validation for tuning, helps stabilize the scoring
    normalize = False,
    rank=(1, 200, 40)       # rank range (start, stop, step) DMD hyperparameter
)
```

```python
experiment_results['tuned_model'].A
```

```python

```
