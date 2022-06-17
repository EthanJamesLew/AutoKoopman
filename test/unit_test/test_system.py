import autokoopman.benchmark.fhn as afhn
import numpy as np


def test_continuous():
    fhn = afhn.FitzHughNagumo()
    training_ivs = np.random.random((10, 2))

    # case test uniform time span
    ret_single = fhn.solve_ivp(training_ivs[0], tspan=(0.0, 3.0), sampling_period=0.05)
    ret_batch = fhn.solve_ivps(training_ivs, tspan=(0.0, 3.0), sampling_period=0.05)
    assert ret_batch.n_trajs == len(training_ivs)

    # case test teval
    teval = np.linspace(0.0, 10.0, 20)
    ret_batch = fhn.solve_ivps(training_ivs, teval=teval, sampling_period=0.05)
    ret_single = fhn.solve_ivp(training_ivs[0], teval=teval, sampling_period=0.05)
    assert len(ret_single.times) == len(teval)
    assert ret_batch.n_trajs == len(training_ivs)


def test_continuous_inputs():
    pass


def test_discrete():
    pass


def test_discrete_inputs():
    pass
