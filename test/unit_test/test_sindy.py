import autokoopman.estimator.sindy as asindy
import autokoopman.benchmark.fhn as afhn
import numpy as np


def test_sindy():
    fhn = afhn.FitzHughNagumo()

    training_ivs = np.random.random((10, 2))
    training_data = fhn.solve_ivps(training_ivs, [0.0, 3.0], 0.05)

    sindy = asindy.PolynomialSindy(["x0", "x1"], (0.05, 2.0))
    sindy_model = sindy.get_model([0.1, 2])
    sindy_model.fit(training_data)