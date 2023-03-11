"""
AutoKoopman hyperparameter tuning module.

The `autokoopman.tuner` module provides classes for hyperparameter tuning of the AutoKoopman algorithms using different optimization techniques. The supported tuners are:
  - GridSearchTuner: Exhaustive grid search over hyperparameter space.
  - MonteCarloTuner: Random search over hyperparameter space using Monte Carlo sampling.
  - BayesianOptTuner: Bayesian optimization of the objective function using GPy.
"""
