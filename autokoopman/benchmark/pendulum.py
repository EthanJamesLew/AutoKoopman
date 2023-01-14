import autokoopman.core.system as asys
import sympy as sp
import numpy as np


class PendulumWithInput(asys.SymbolicContinuousSystem):
    r"""
    Simple Pendulum with Constant Torque Input
        We model a pendulum with the equation:

        .. math::
            l^2 \ddot{\theta} + \beta \theta + g l \sin \theta = \tau

        This leads to the state space form:
        
        .. math::
            \left\{\begin{array}{l}
            \dot{x}_{1} = x_2 \\
            \dot{x}_{2} = - g / l \sin x_1 - 2 \beta x_2 + u_1 \\
            \end{array}\right.

        Note that :math:`\beta=b / m` kn this formulation: http://underactuated.mit.edu/pend.html .
    """

    def __init__(self, g=9.81, l=1.0, beta=0.0):
        self.name = "pendulum"
        self.init_set_low = [-1, -1]
        self.init_set_high = [1, 1]
        self.input_type = "step"
        self.teval = np.linspace(0, 10, 200)
        self.input_set_low = [-1]
        self.input_set_high = [1]
        theta, thetadot = sp.symbols("theta thetadot")
        tau = sp.symbols("tau")
        xdot = [thetadot, -g / l * sp.sin(theta) - 2 * beta * thetadot + tau]
        super(PendulumWithInput, self).__init__(
            (theta, thetadot), xdot, input_variables=(tau,)
        )
