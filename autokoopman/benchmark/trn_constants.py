import sympy as sp  # type: ignore
import autokoopman.core.system as asys
import numpy as np


class TRNConstants(asys.SymbolicContinuousSystem):
    r"""
    Oscillatory network of transcriptional regulators with N genes

    We treat the unknown inputs as variables with xdot set to zero such that they are given an initial range.

    References
        Maiga, M., Ramdani, N., Travé-Massuyès, L. and Combastel, C., 2015. A comprehensive method for reachability analysis of uncertain nonlinear hybrid systems.
        IEEE Transactions on Automatic Control, 61(9), pp.2341-2356.
    """

    def __init__(self, b=-0.5, n=2, alpha=1, k=2, mu =2, alpha_t = 25.5, n_genes=24):
        self.name = "trn_constants"
        init_set_low = [35, 27, 30, 25, 40, 32]
        init_set_high = [40, 32, 35, 30, 45, 37]
        self.init_set_low = []
        self.init_set_high = []
        for i in range(n_genes):
            self.init_set_low += [init_set_low[(i%3)*2], init_set_low[(i%3)*2+1]]
            self.init_set_high += [init_set_high[(i%3)*2], init_set_high[(i%3)*2+1]]
        trn_vars_string = ""
        for i in range(1, n_genes + 1):
            trn_vars_string += f"m{i} p{i} "
        trn_vars = sp.symbols(trn_vars_string)
        xdot = []
        for i in range(n_genes):
            xdot += [
                -trn_vars[2 * i] + b * trn_vars[2 * i + 1] + (alpha/ (1 + trn_vars[2 * (i - 1) + 1] ** n)) +
                alpha_t, k * trn_vars[2 * i] - mu * trn_vars[2 * i + 1]]
        super(TRNConstants, self).__init__(trn_vars, xdot)
