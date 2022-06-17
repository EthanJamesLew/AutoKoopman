import autokoopman.core.system as asys
import sympy as sp


class PendulumWithInput(asys.SymbolicContinuousSystem):
    r"""
    TODO: comment this
    """

    def __init__(self, g=9.81, l=1.0, beta=0.0):
        theta, thetadot = sp.symbols("theta thetadot")
        tau = sp.symbols("tau")
        xdot = [thetadot, -g / l * sp.sin(theta) - 2 * beta * thetadot + tau]
        super(PendulumWithInput, self).__init__(
            (theta, thetadot), xdot, input_variables=(tau,)
        )
