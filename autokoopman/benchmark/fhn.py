import sympy as sp  # type: ignore

import autokoopman.core.system as asys


class FitzHughNagumo(asys.SymbolicContinuousSystem):
    r"""
    FitzHugh-Nagumo System (FHN)
        The FHN Model is an example of a relaxation oscillator.

        TODO: parameterize this to a more general set of equations.

        We implement the system in the specific case.

        .. math::
            \left\{\begin{array}{l}
            \dot{x}_{1}=3 (x_1 - x_1^3 / 3 + x_2) \\
            \dot{x}_{2}=(0.2 - 3 x_1 - 0.2 x_2) / 0.3 \\
            \end{array}\right.

    References
        FitzHugh, R. (1961). Impulses and physiological states in theoretical models of nerve membrane.
        Biophysical journal, 1(6), 445-466.
    """

    def __init__(self):
        x0, x1 = sp.symbols("x0 x1")
        xdot = [3 * (x0 - x0**3 / 3 + x1), (0.2 - 3 * x0 - 0.2 * x1) / 0.3]
        super(FitzHughNagumo, self).__init__((x0, x1), xdot)
