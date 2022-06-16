import pytest


def test_trajectory():
    import numpy as np
    import autokoopman.core.trajectory as traj

    states = np.random.random((10, 2))
    times = np.linspace(0.0, 2.0, 10)
    names = [f"x{idx}" for idx in range(2)]
    with pytest.raises(AssertionError) as excinfo:
        traj = traj.Trajectory(times, states, states, ['wrong name'])

    traj = traj.Trajectory(times, states, None, names, None, threshold=0.01)
    assert len(traj.names) == 2

    traj.interp1d(np.linspace(0.0, 2.0, 5))

    traj.norm()

    traj.interp_uniform_time(0.1)

    assert traj.dimension == 2

    assert traj.states.shape[0] == 10
    assert traj.states.shape[1] == 2


def test_input_trajectory():
    import numpy as np
    import autokoopman.core.trajectory as traj

    states = np.random.random((10, 2))
    times = np.linspace(0.0, 2.0, 10)
    names = [f"x{idx}" for idx in range(2)]

    with pytest.raises(AssertionError) as excinfo:
        traj = traj.Trajectory(times, states, states, names, ['wrong name'])

