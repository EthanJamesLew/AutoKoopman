import sympy as sp
import autokoopman.core.system as asys


class VanderPol(asys.SymbolicContinuousSystem):
    r"""
    Coupled VanderPol
        A two-dimensional continuous model a Van der Pol oscillator [1].

        Initial set:
            x ∈ [1.25,1.55]
            y ∈ [2.25,2.35]

    References
[1] https://ths.rwth-aachen.de/research/projects/hypro/van-der-pol-oscillator/
 """

    def __init__(self, mu=1):
        self.name = "coupledVanderPol"
        self.init_set_low = [1.25, 2.25]
        self.init_set_high = [1.55, 2.35]
        x, y = sp.symbols("x y")
        xdot = [
            y,
            mu*(1.0 - x ** 2) * y - x,
        ]
        super(VanderPol, self).__init__((x, y), xdot)
