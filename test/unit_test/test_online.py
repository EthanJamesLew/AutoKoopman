"""
Tests for autokoopman.estimator.online_koopman

Test with and without inputs, with and with normalization
"""
import pytest
import autokoopman.estimator.online_koopman as okoop
import autokoopman.core.observables as obs

import matplotlib.pyplot as plt
import numpy as np


koop_config = ((True,), (False,))


@pytest.mark.parametrize(
    "normalize",
    koop_config,
)
def test_online(normalize):
    # initialize on short bursts
    import autokoopman.benchmark.fhn as fhn

    fhn = fhn.FitzHughNagumo()
    iv = [0.5, 0.1]
    training_data = fhn.solve_ivps(
        initial_states=np.random.uniform(low=-2.0, high=2.0, size=(1, 2)),
        tspan=[0.0, 10.0],
        sampling_period=0.1,
    )
    # create an online estimator
    online = okoop.OnlineKoopmanEstimator(
        obs.IdentityObservable() | obs.RFFObservable(2, 25, 0.5), normalize=True
    )
    online.initialize(training_data)

    for _ in range(3):
        streaming_data = fhn.solve_ivps(
            initial_states=np.random.uniform(low=-2.0, high=2.0, size=(1, 2)),
            tspan=[0.0, 1.0],
            sampling_period=0.1,
        )
        # update the online estimator
        online.update(streaming_data)

        # check that we can simulate the model
        online.model.solve_ivp(initial_state=iv, tspan=(0.0, 10.0), sampling_period=0.1)


@pytest.mark.parametrize(
    "normalize",
    koop_config,
)
def test_online_with_inputs(normalize):
    from autokoopman.benchmark.pendulum import PendulumWithInput

    # create the pendulum system
    pendulum_sys = PendulumWithInput(beta=0.05)
    iv = [0.5, 0.1]

    def make_input_step(duty, on_amplitude, off_amplitude, teval):
        """produce a step response input signal for the pendulum"""
        length = len(teval)
        on_amplitude *= 4.0
        on_amplitude -= 2.0
        off_amplitude *= 4.0
        off_amplitude -= 2.0
        inp = np.zeros((length,))
        phase_idx = int(length * duty)
        inp[:phase_idx] = on_amplitude
        inp[phase_idx:] = off_amplitude
        return inp

    #  training data
    teval = np.linspace(0, 3, 200)
    dt = 10 / 200.0
    params = np.random.rand(10, 3)
    ivs = np.random.uniform(low=-1.0, high=1.0, size=(10, 2))
    steps = [make_input_step(*p, teval) for p in params]
    training_data = pendulum_sys.solve_ivps(ivs, inputs=steps, teval=teval)
    test_inputs = np.atleast_2d(make_input_step(*np.random.rand(3), teval)).T

    # create an online estimator
    online = okoop.OnlineKoopmanEstimator(
        obs.IdentityObservable() | obs.RFFObservable(2, 25, 0.5), normalize=False
    )
    online.initialize(training_data.interp_uniform_time(dt))

    for _ in range(3):
        streaming_data = pendulum_sys.solve_ivps(
            initial_states=np.random.uniform(low=-2.0, high=2.0, size=(1, 2)),
            teval=teval,
            inputs=[make_input_step(*p, teval) for p in np.random.rand(1, 3)],
        )
        # update the online estimator
        online.update(streaming_data.interp_uniform_time(dt))

        # check that we can simulate the model
        online.model.solve_ivp(initial_state=iv, teval=teval, inputs=test_inputs)
