import sympy as sp  # type: ignore

import autokoopman.core.system as asys


class FitzHughNagumo(asys.SymbolicContinuousSystem):
    r"""
    FitzHugh-Nagumo System (FHN)
        The FHN Model is an example of a relaxation oscillator.

        .. math::
            \left\{\begin{array}{l}
            \dot{x}_{1}=x_0 - \frac{x_0^3}{3} - x_1 + R I_{ext} \\
            \dot{x}_{2}=\frac{x_0 + a - b x_1}{\tau} \\
            \end{array}\right.

    :param i_ext: external stimulus
    :param r: resistance
    :param a: model parameter :math:`a`
    :param b: model parameter :math:`b`
    :param tau: model parameter :math:`\tau`

    Setting :math:`a=b=0` creates the Van der Pol Oscillator.

    Example:
        .. code-block:: python

            import autokoopman.benchmark.fhn as fhn

            sys = fhn.FitzHughNagumo()
            traj = sys.solve_ivp(
                initial_state=[2.0, 1.5],
                tspan=(0.0, 20.0),
                sampling_period=0.01
            )
            traj.states
            # array([[ 2.        ,  1.5       ],
            #        [ 1.97862133,  1.50612782],
            #        [ 1.95779684,  1.51217922],
            #        ...,
            #        [-1.32849614, -0.81944446],
            #        [-1.32576847, -0.82204749],
            #        [-1.32303553, -0.82463882]])

    References
        FitzHugh, R. (1961). Impulses and physiological states in theoretical models of nerve membrane.
        Biophysical journal, 1(6), 445-466.
    """

    def __init__(self, i_ext=0.0, r=1.0, a=0.3, b=0.3, tau=3.0):
        x0, x1 = sp.symbols("x0 x1")
        xdot = [x0 - x0**3 / 3.0 - x1 + r * i_ext, (x0 + a - b * x1) / tau]
        super(FitzHughNagumo, self).__init__((x0, x1), xdot)
