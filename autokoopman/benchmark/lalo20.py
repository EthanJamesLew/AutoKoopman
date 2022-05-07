import sympy as sp  # type: ignore

import autokoopman.core.system as asys

"""
They are boxes centered at x1(0) = 1.2, x2(0) = 1.05, x3(0) = 1.5, x4(0) = 2.4, x5(0) = 1, x6(0) = 0.1, x7(0) = 0.45.
The range of the box in the i-th dimension is defined by the interval [xi(0) − W, xi(0) + W ].

Weconsider W = 0.01, W = 0.05, and W =0.1. For W =0.01 and W =0.05 we consider the unsafe region defined by x4 ≥ 4.5,
while for W = 0.1, the unsafe set is defined by x4 ≥ 5. The time horizon for all cases is [0, 20].

For more details see: https://easychair.org/publications/open/nrdD

"""


class LaubLoomis(asys.SymbolicContinuousSystem):
    def __init__(self):
        x1, x2, x3, x4, x5, x6, x7 = sp.symbols("x1 x2 x3 x4 x5 x6 x7")
        xdot = [
            1.4 * x3 - 0.9 * x1,
            2.5 * x5 - 1.5 * x2,
            0.6 * x7 - 0.8 * x2 * x3,
            2 - 1.3 * x3 * x4,
            0.7 * x1 - x4 * x5,
            0.3 * x1 - 3.1 * x6,
            1.8 * x6 - 1.5 * x2 * x7,
        ]
        super(LaubLoomis, self).__init__((x1, x2, x3, x4, x5, x6, x7), xdot)
