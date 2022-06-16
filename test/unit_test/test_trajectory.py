import pytest


def test_trajectory():
    import numpy as np
    import autokoopman.core.trajectory as traj

    # generate good and bad time state pairs
    states = np.random.random((10, 2))
    times = np.linspace(0.0, 2.0, 10)
    times_bad = np.linspace(0.0, 2.0, 8)

    # good state name
    names = [f"x{idx}" for idx in range(2)]

    # check that bad name fails
    with pytest.raises(AssertionError) as excinfo:
        traj = traj.Trajectory(times, states, states, ["wrong name"])

    # check that bad time (wrong number) fails
    with pytest.raises(AssertionError) as excinfo:
        traj_b = traj.Trajectory(times_bad, states, states)

    # good trajectory case
    traj = traj.Trajectory(times, states, None, names, None, threshold=0.01)
    assert len(traj.names) == 2

    # call some of the methods and see that they don't produce errors
    # TODO: test these better
    traj.interp1d(np.linspace(0.0, 2.0, 5))
    traj.norm()
    traj.interp_uniform_time(0.1)

    # check some properties
    assert traj.dimension == 2
    assert traj.states.shape[0] == 10
    assert traj.states.shape[1] == 2


def test_input_trajectory():
    import numpy as np
    import autokoopman.core.trajectory as traj

    # case: make good and bad state input pairs
    states = np.random.random((10, 2))
    inputs = np.random.random((10, 3))
    inputs_bad = np.random.random((7, 2))

    times = np.linspace(0.0, 2.0, 10)
    names_inputs = [f"x{idx}" for idx in range(3)]
    names = [f"x{idx}" for idx in range(2)]

    # case bad names
    with pytest.raises(AssertionError) as excinfo:
        traj_b = traj.Trajectory(times, states, inputs, names, ["wrong name"])

    with pytest.raises(AssertionError) as excinfo:
        traj_b = traj.Trajectory(times, states, inputs_bad, names, ["wrong name"])

    # create a good trajectory
    traj_g = traj.Trajectory(times, states, inputs, names, names_inputs)

    # call some of the methods and see that they don't produce errors
    # TODO: test these better
    traj_i = traj_g.interp1d(np.linspace(0.0, 2.0, 5))
    traj_g.norm()
    traj_g.interp_uniform_time(0.1)

    # check some properties
    for tr in {traj_i, traj_g}:
        assert tr.input_dimension == 3
        assert tr.dimension == 2
        assert tr.states.shape[0] in {5, 10}
        assert tr.states.shape[1] == 2
