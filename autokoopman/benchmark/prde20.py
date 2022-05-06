import sympy as sp  # type: ignore

import autokoopman.core.system as asys

"""
In this model, x0 is the nutrients, x1 the phytoplankton and x2 the detritus.
The constraints are x0(t), x1(t), x2(t) are positive and x0(t) + x1(t) + x2(t) = 10 for all t.

The initial condition x0(0) = 9.98, x1(0) = 0.01 and x2(0) = 0.01

I: x0(0) ∈ [9.5,10.0], i.e., uncertainty on the initial condition;

For more details see: https://easychair.org/publications/open/nrdD

"""
class ProdDestr(asys.SymbolicContinuousSystem):
    def __init__(self):
        x0, x1, x2 = sp.symbols("x0 x1 x2")
        xdot = [(-x0 * x1) / (1 + x0), (x0 * x1) / (1 + x0) - 0.3*x1,0.3*x1]
        super(ProdDestr, self).__init__((x0, x1, x2), xdot)
