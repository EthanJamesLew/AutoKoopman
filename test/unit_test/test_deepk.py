def test_deepk():
    """test that deep koopman runs"""
    # the notebook imports
    import numpy as np
    import autokoopman.benchmark.fhn as fhn

    # make a training and validation set
    dt = 0.1
    fhn = fhn.FitzHughNagumo()
    training_data = fhn.solve_ivps(
        initial_states=np.random.uniform(low=-4.0, high=4.0, size=(100, 2)),
        tspan=[0.0, 3.0],
        sampling_period=dt,
    )

    test_ivs = np.random.uniform(low=-2.0, high=2.0, size=(10, 2))
    test_data = fhn.solve_ivps(
        initial_states=test_ivs, tspan=[0.0, 10.0], sampling_period=dt
    )

    # run deepk
    import autokoopman.estimator.deepkoopman as dk

    koop = dk.DeepKoopman(
        2,
        0,
        40,
        max_iter=10,  # very low count for bring up test
        lr=1e-3,
        hidden_enc_dim=64,
        num_hidden_layers=2,
        metric_loss_weight=0.3,
        pred_loss_weight=20.0,
        torch_device="cpu",  #  don't require a GPU!
        validation_data=test_data,
    )
    koop.fit(training_data)
