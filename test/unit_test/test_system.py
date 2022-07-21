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
    from autokoopman.benchmark.pendulum import PendulumWithInput

    iv = np.array([0.1, 0.0])
    teval = np.linspace(0.0, 5.0, 50)
    inputs = np.zeros(teval.shape)
    inputs[25] = 1.0
    pend = PendulumWithInput()
    sol = pend.solve_ivp(iv, teval=teval, inputs=inputs)


def test_discrete():
    from autokoopman.core.system import StepDiscreteSystem

    step = lambda t, s, i: np.array([s[0] - 0.1 * s[1], s[1] + 0.01 * s[0]])
    my_sys = StepDiscreteSystem(step, ["x0", "x1"])
    my_sys.solve_ivp(np.array([0.1, 0.1]), tspan=(0.0, 1.0), sampling_period=0.05)


def test_discrete_inputs():
    from autokoopman.core.system import StepDiscreteSystem

    step = lambda t, s, i: np.array([s[0] - 0.1 * s[1], s[1] + 0.01 * s[0] + i[0]])
    my_sys = StepDiscreteSystem(step, ["x0", "x1"])
    teval = np.linspace(0.0, 2.0, 20)
    inputs = teval
    my_sys.solve_ivp(
        np.array([0.1, 0.1]), teval=teval, inputs=inputs, sampling_period=0.05
    )
