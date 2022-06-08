import sympy as sp  # type: ignore

import autokoopman.core.system as asys


class Bio2(asys.SymbolicContinuousSystem):
    r"""
    Spring pendulum
        A nine-dimensional continuous model which is adapted from a biological system given in [1].

        Initial set:
            x1 in [0.99,1.01]
            x2 in [0.99,1.01]
            x3 in [0.99,1.01]
            x4 in [0.99,1.01]
            x5 in [0.99,1.01]
            x6 in [0.99,1.01]
            x7 in [0.99,1.01]
            x8 in [0.99,1.01]
            x9 in [0.99,1.01]

    References
         E. Klipp, R. Herwig, A. Kowald, C. Wierling, H. Lehrach. Systems Biology in Practice: Concepts, Implementation and Application. Wiley-Blackwell, 2005.
    """

    def __init__(self):
        x1, x2, x3, x4, x5, x6, x7, x8, x9 = sp.symbols("x1 x2 x3 x4 x5 x6 x7 x8 x9   ")
        xdot = [
            3 * x3 - x1 * x6,
            x4 - x2 * x6,
            x1 * x6 - 3 * x3,
            x2 * x6 - x4,
            3 * x3 + 5 * x1 - x5,
            5 * x5 + 3 * x3 + x4 - x6 * (x1 + x2 + 2 * x8 + 1),
            5 * x4 + x2 - 0.5 * x7,
            5 * x7 - 2 * x6 * x8 + x9 - 0.2 * x8,
            2 * x6 * x8 - x9,
        ]
        super(Bio2, self).__init__((x1, x2, x3, x4, x5, x6, x7, x8, x9), xdot)
