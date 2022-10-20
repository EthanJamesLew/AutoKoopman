import sympy as sp  # type: ignore
import autokoopman.core.system as asys


class TRN(asys.SymbolicContinuousSystem):
    r"""
    Oscillatory network of transcriptional regulators with N genes

    We treat the unknown inputs as variables with xdot set to zero such that they are given an initial range.

    References
        Maiga, M., Ramdani, N., Travé-Massuyès, L. and Combastel, C., 2015. A comprehensive method for reachability analysis of uncertain nonlinear hybrid systems.
        IEEE Transactions on Automatic Control, 61(9), pp.2341-2356.
    """

    def __init__(self):
        self.name = "trn"
        # self.init_set_low = [1.1, 0.95, 1.4, 2.3, 0.9, 0, 0.35]
        # self.init_set_high = [1.3, 1.15, 1.6, 2.5, 1.1, 0.2, 0.55]
        trn_vars_string = "alpha"
        for i in range(1, 25): vars_string += f"m{i} k{i} alpha{i}"
        print(trn_vars_string)
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
        super(TRN, self).__init__((x1, x2, x3, x4, x5, x6, x7), xdot)
