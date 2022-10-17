import numpy as np


def test_discrete_koopman():
    """tests the discrete time Koopman estimator"""
    from autokoopman.benchmark.pendulum import PendulumWithInput
    from autokoopman.core.trajectory import TrajectoriesData
    from autokoopman.core.observables import IdentityObservable, RFFObservable
    from autokoopman.estimator.koopman import KoopmanDiscEstimator
    import random

    pend = PendulumWithInput(l=0.6, beta=0.1)
    n_trajs = 40
    training_ivs = np.random.random((n_trajs, 2))
    trajs = {}
    teval = np.linspace(0.0, 5.0, 100)
    for i in range(n_trajs):
        inputs = (
            random.random() * 2.0 * np.cos(random.random() * teval + random.random())
        )
        sol = pend.solve_ivp(training_ivs[i], teval=teval, inputs=inputs)
        trajs[i] = sol
    sols = TrajectoriesData(trajs)
    sols = sols.interp_uniform_time(1 / 20.0)
    disc = KoopmanDiscEstimator(
        IdentityObservable() | RFFObservable(2, 100, 0.001), 0.1, 2, 100
    )
    disc.fit(sols)
    new_inputs = inputs.copy()
    new_inputs[-len(inputs) // 2 :] = -10.0
    new_inputs[: len(inputs) // 2] = 0.0
    preds = disc.model.solve_ivps(
        training_ivs,
        teval=teval,
        inputs=[new_inputs for _ in range(len(training_ivs))],
        sampling_period=1 / 20.0,
    )
    new_sols = pend.solve_ivps(
        training_ivs,
        teval=teval,
        inputs=[new_inputs for _ in range(len(training_ivs))],
        sampling_period=1 / 20.0,
    )


def test_cont_koopman():
    """tests the continuous time Koopman estimator"""
    from autokoopman.benchmark.pendulum import PendulumWithInput
    from autokoopman.core.trajectory import TrajectoriesData
    from autokoopman.core.observables import IdentityObservable, RFFObservable
    from autokoopman.estimator.koopman import KoopmanContinuousEstimator
    import random

    pend = PendulumWithInput(l=0.6, beta=0.1)
    n_trajs = 40
    training_ivs = np.random.random((n_trajs, 2))
    trajs = {}
    teval = np.linspace(0.0, 5.0, 100)
    for i in range(n_trajs):
        inputs = (
            random.random() * 2.0 * np.cos(random.random() * teval + random.random())
        )
        sol = pend.solve_ivp(training_ivs[i], teval=teval, inputs=inputs)
        trajs[i] = sol
    sols = TrajectoriesData(trajs)
    sols = sols.interp_uniform_time(1 / 20.0)
    cont = KoopmanContinuousEstimator(
        IdentityObservable() | RFFObservable(2, 100, 0.001), 2, 100
    )
    cont.fit(sols)
    new_inputs = inputs.copy()
    new_inputs[-len(inputs) // 2 :] = -10.0
    new_inputs[: len(inputs) // 2] = 0.0
    preds = cont.model.solve_ivps(
        training_ivs,
        teval=teval,
        inputs=[new_inputs for _ in range(len(training_ivs))],
        sampling_period=1 / 20.0,
    )
    new_sols = pend.solve_ivps(
        training_ivs,
        teval=teval,
        inputs=[new_inputs for _ in range(len(training_ivs))],
        sampling_period=1 / 20.0,
    )
