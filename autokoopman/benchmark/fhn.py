import autokoopman.core.system as asys
import sympy as sp


class FitzHughNagumo(asys.SymbolicContinuousSystem):
    def __init__(self):
        x0, x1 = sp.symbols('x0 x1')
        xdot = [3 * (x0 - x0 ** 3 / 3 + x1), (0.2 - 3 * x0 - 0.2 * x1) / 0.3]
        super(FitzHughNagumo, self).__init__((x0, x1), xdot)
