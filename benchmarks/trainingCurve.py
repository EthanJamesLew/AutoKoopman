"""
Figure for Tuning Scores
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import autokoopman.benchmark.lalo20 as lalo20
import autokoopman as ak

if __name__ == "__main__":
    # set the seeds
    random.seed(1234)
    np.random.seed(1234)

    # for a complete example, let's create an example dataset using an included benchmark system
    print("generating training data...")
    fhn = lalo20.LaubLoomis()
    training_data = fhn.solve_ivps(
        initial_states=np.random.uniform(low=0.1, high=5.0, size=(30, 7)),
        tspan=(0.0, 10.0),
        sampling_period=0.1
    )

    # learn model from data
    print("running tuners...")
    opts = ["bopt", "monte-carlo", "grid"]
    tuners = {}
    for tuner_name in opts:
        experiment_results = ak.auto_koopman(
            training_data,          # list of trajectories
            sampling_period=0.1,    # sampling period of trajectory snapshots
            obs_type="rff",         # use Random Fourier Features Observables
            opt=tuner_name,       
            n_obs=200,              # maximum number of observables to try
            max_opt_iter=200,       # maximum number of optimization iterations
            grid_param_slices=10,   # for grid search, number of slices for each parameter
            n_splits=5,             # k-folds validation for tuning, helps stabilize the scoring
            rank=(1, 200, 10)       # rank range (start, stop, step) DMD hyperparameter
        )
        tuners[tuner_name] = experiment_results['tuner']

    plt.figure(figsize=(8*0.8, 5*0.8))
    pretty_name = {
        "bopt": "Bayesian Optimization",
        "grid": "Grid Search",
        "monte-carlo": "Random Search"
        
    }

    # plot the result
    print("plotting results...")
    for tuner_name, tuner in tuners.items():
        plt.plot(tuner.best_scores, label=pretty_name[tuner_name])
    plt.xlabel("iteration")
    plt.ylabel("Error")
    plt.yscale('log')
    plt.legend()
    plt.grid()
    print("saving tuning_error.pdf")
    plt.savefig("tuning_error.pdf")

    # save also a pdf
    import pandas as pd
    columns = list(tuners.keys())
    data = pd.DataFrame(
        columns=columns,
        data=np.array([tuners[tname].best_scores for tname in columns]).T
    )
    data.to_csv("tuning_error.csv")

