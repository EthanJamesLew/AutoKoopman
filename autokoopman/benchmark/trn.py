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

    # Wrong, we need to have alpha, k and mu as inputs, what shape would they be??

    def __init__(self, b=-0.5, n=2, n_genes=24):
        self.name = "trn"
        init_pattern_low = [35, 27, 30, 25, 40, 32]
        init_pattern_high = [40, 32, 35, 30, 45, 37]
        self.init_set_low = [0.5, 1.9, 1.9]
        self.init_set_high = [1.5, 2.1, 2.1]
        for i in range(n_genes):
            self.init_set_low += [init_pattern_low[(i % 3) * 2], init_pattern_low[(i % 3) * 2 + 1], 25]
            self.init_set_high += [init_pattern_high[(i % 3) * 2], init_pattern_high[(i % 3) * 2 + 1], 26]
        trn_vars_string = "alpha k mu "
        for i in range(1, n_genes+1): trn_vars_string += f"m{i} p{i} alpha{i} "
        trn_vars = sp.symbols(trn_vars_string)
        xdot = [0, 0, 0]
        xdot += [-trn_vars[3] + b * trn_vars[4] + (trn_vars[0] / (1 + trn_vars[3 * 24 + 1] ** n)) + trn_vars[5],
                 trn_vars[1] * trn_vars[3] - trn_vars[2] * trn_vars[4], 0]
        for i in range(2, n_genes+1): xdot += [
            -trn_vars[3 * i] + b * trn_vars[3 * i + 1] + (trn_vars[0] / (1 + trn_vars[3 * (i - 1) + 1] ** n)) +
            trn_vars[3 * i + 2],
            trn_vars[1] * trn_vars[3 * i] - trn_vars[2] * trn_vars[3 * i + 1], 0]
        super(TRN, self).__init__(trn_vars, xdot)
