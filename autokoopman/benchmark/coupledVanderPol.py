import sympy as sp
import autokoopman.core.system as asys


class CoupledVanderPol(asys.SymbolicContinuousSystem):
    r"""
    Coupled VanderPol
        A four-dimensional continuous model of two coupled Van der Pol oscillators [1, 2].

        Initial set:
            x1 ∈ [1.25,1.55]
            y1 ∈ [2.25,2.35]
            x2 ∈ [1.25,1.55]
            y2 ∈ [2.25,2.35]


    References
[1] https://ths.rwth-aachen.de/research/projects/hypro/coupled-van-der-pol-oscillator/
[2] R. H. Rand and P. J. Holmes. Bifurcation of periodic motions in two weakly coupled Van der Pol oscillators. Volume 15 of International Journal of Non-Linear Mechanics, pages 387–399, Pergamon Press Ltd., 1980.
 """

    def __init__(self, g=9.81):
        self.name = "coupledVanderPol"
        self.init_set_low = [1.25, 2.25, 1.25, 2.25]
        self.init_set_high = [1.55, 2.35, 1.55, 2.35]
        x1, y1, x2, y2 = sp.symbols("x1 y1 x2 y2")
        xdot = [
            y1,
            (1.0 - x1 ** 2) * y1 - x1 + (x2 - x1),
            y2,
            (1.0 - x2 ** 2) * y2 - x2 + (x1 - x2),
        ]
        super(CoupledVanderPol, self).__init__((x1, y1, x2, y2), xdot)
