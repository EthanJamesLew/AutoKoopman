import sympy as sp  # type: ignore
import autokoopman.core.system as asys
import numpy as np


class TRNInput(asys.SymbolicContinuousSystem):
    r"""
    Oscillatory network of transcriptional regulators with N genes

    We treat the unknown inputs as variables with xdot set to zero such that they are given an initial range.

    References
        Maiga, M., Ramdani, N., Travé-Massuyès, L. and Combastel, C., 2015. A comprehensive method for reachability analysis of uncertain nonlinear hybrid systems.
        IEEE Transactions on Automatic Control, 61(9), pp.2341-2356.
    """

    def __init__(self, b=-0.5, n=2, n_genes=24):
        self.name = "trn_inputs"
        init_set_low = [35, 27, 30, 25, 40, 32]
        init_set_high = [40, 32, 35, 30, 45, 37]
        self.init_set_low = []
        self.init_set_high = []
        self.input_set_low = [0.5, 1.9, 1.9]
        self.input_set_high = [1.5, 2.1, 2.1]
        self.input_type = "rand"
        self.teval = np.linspace(0, 10, 200)
        for i in range(n_genes):
            self.init_set_low += [init_set_low[(i%3)*2], init_set_low[(i%3)*2+1]]
            self.init_set_high += [init_set_high[(i%3)*2], init_set_high[(i%3)*2+1]]
            self.input_set_low.append(25)
            self.input_set_high.append(26)
        trn_vars_string = ""
        input_vars_string = "alpha k mu"
        for i in range(1, n_genes + 1):
            trn_vars_string += f"m{i} p{i} "
            input_vars_string += f" alpha{i}"
        trn_vars = sp.symbols(trn_vars_string)
        input_vars = sp.symbols(input_vars_string)
        xdot = []
        for i in range(n_genes):
            xdot += [
                -trn_vars[2 * i] + b * trn_vars[2 * i + 1] + (input_vars[0] / (1 + trn_vars[2 * (i - 1) + 1] ** n)) +
                input_vars[i + 3], input_vars[1] * trn_vars[2 * i] - input_vars[2] * trn_vars[2 * i + 1]]
        super(TRNInput, self).__init__(trn_vars, xdot, input_variables=input_vars)
