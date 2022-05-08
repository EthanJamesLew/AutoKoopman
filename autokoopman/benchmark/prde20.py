import sympy as sp  # type: ignore

import autokoopman.core.system as asys


class ProdDestr(asys.SymbolicContinuousSystem):
    r"""
    Production Destruction System (PRDE20)
        In this model, :math:`x_0` is the nutrients, :math:`x_1` the phytoplankton and :math:`x_2` the detritus.
        The constraints are :math:`x_0(t), x_1(t), x_2(t)` are positive and :math:`x_0(t) + x_1(t) + x_2(t) = 10` for all t.

        The initial condition :math:`x_0(0) = 9.98`, :math:`x_1(0) = 0.01` and :math:`x_2(0) = 0.01`

        :math:`I: x_0(0) \in [9.5, 10.0]`, i.e., uncertainty on the initial condition;

        From Michaelis-Menten Theory, the evolution function is

        .. math::
            \begin{bmatrix} \dot x_0 \\ \dot x_1 \\ \dot x_2 \end{bmatrix} = \begin{bmatrix} \frac{-x_0 x_1}{1 + x_0} \\ \frac{x_0 x_1}{1 + x_0} - 0.3 x_1 \\ 0.3 x_1 \end{bmatrix}

    :param alpha: model parameter

    Example:
        .. code-block:: python

            import autokoopman.benchmark.prde20 as prde

            sys = prde.ProdDestr()
            traj = sys.solve_ivp(
                initial_state=[9.98, 0.01, 0.01],
                tspan=(0.0, 20.0),
                sampling_period=0.01
            )
            traj.states
            # array([[9.98000000e+00, 1.00000000e-02, 1.00000000e-02],
            #        [9.97990883e+00, 1.00610783e-02, 1.00300915e-02],
            #        [9.97981710e+00, 1.01225295e-02, 1.00603668e-02],
            #        ...,
            #        [3.34608394e-08, 4.41976926e-01, 9.55802304e+00],
            #        [3.33135031e-08, 4.40652987e-01, 9.55934698e+00],
            #        [3.31672496e-08, 4.39333011e-01, 9.56066696e+00]])

    References
        Geretti, L., Sandretto, J. A. D., Althoff, M., Benet, L., Chapoutot, A., Chen, X., ... & Schilling, C. (2020).
        ARCH-COMP20 category report: Continuous and hybrid systems with nonlinear dynamics. EPiC Series in Computing, 74, 49-75. pp 53
    """

    def __init__(self, alpha=0.3):
        x0, x1, x2 = sp.symbols("x0 x1 x2")
        xdot = [(-x0 * x1) / (1 + x0), (x0 * x1) / (1 + x0) - alpha * x1, alpha * x1]
        super(ProdDestr, self).__init__((x0, x1, x2), xdot)
