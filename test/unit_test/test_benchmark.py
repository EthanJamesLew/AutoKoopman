import autokoopman.benchmark.fhn as afhn
import numpy as np


def test_fhn():
    fhn = afhn.FitzHughNagumo()
    training_ivs = np.random.random((10, 2))
    fhn.solve_ivps(training_ivs, (0.0, 3.0), sampling_period=0.05)
