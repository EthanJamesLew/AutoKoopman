import sympy as sp  # type: ignore

import autokoopman.core.system as asys


class RobBench(asys.SymbolicContinuousSystem):
    r"""
    Robertson chemical reaction benchmark (ROBE21)
        As proposed by Robertson [31], this chemical reaction system models the kinetics of an auto-catalytic reaction.
        x, y and z are the (positive) concentrations of the species, with the assumption that x+y+z = 1. Here alpha is a small
        constant, while beta and gamma take on large values.
        The initial condition is always :math:`x(0) = 1`, :math:`y(0) = 0` and :math:`z(0) = 0`.

        :math:`I: x_0(0) \in [9.5, 10.0]`, i.e., uncertainty on the initial condition;

        In this benchmark we fix alpha = 0.4 and analyze the system under three different pairs of values for beta and gamma:
        1. beta = 10^2 , gamma = 10^3
        2. beta=10^3, gamma = 10^5
        3. beta = 10^3, gamma = 10^7

        We are interested in computing the reachable tube until t = 40, to see how the integration scheme holds under the stiff behavior. No verification objective is enforced.

        For each of the three setups, the following three measures are required:
        1. the execution time for evolution;
        2. the number of integration steps taken;
        3. the width of the sum of the concentrations s = x + y + z at the final time.
        Additionally, a figure with s (in the [0.999, 1.001] range) w.r.t. time overlaid for the three setups should be shown.

        As proposed by Robertson, this chemical reaction system models the kinetics of an auto- catalytic reaction.

    References
        Geretti, L., dit Sandretto, J. A., Althoff, M., Benet, L., Chapoutot, A., Collins, P., ... & Wetzlinger, M. (2021). ARCH-COMP21 Category Report: Continuous and Hybrid Systems with Nonlinear Dynamics. EPiC Series in Computing, 80, 32-54.
        H. H. Robertson. The solution of a set of reaction rate equations. In ”Numerical analysis: an introduction”, page 178–182. Academic Press, 1966.
    """

    def __init__(self, alpha=0.4, beta=100, gamma=1000):
        x, y, z = sp.symbols("x y z")
        xdot = [
            -alpha * x + beta * y * z,
            alpha * x - beta * y * z - gamma * y * y,
            gamma * y * y,
        ]
        super(RobBench, self).__init__((x, y, z), xdot)
