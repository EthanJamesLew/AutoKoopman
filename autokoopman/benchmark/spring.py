import sympy as sp  # type: ignore

import autokoopman.core.system as asys


class Spring(asys.SymbolicContinuousSystem):
    r"""
    Spring pendulum
        We study the behavior of the planar spring-pendulum described in [1].
        It consists of a solid ball of mass m and a spring of natural length L.
        The spring constant is k.

        We study the evolutions of the length r of the spring and the angle \theta
        between the spring and the vertical. They are modeled by the following differential
        equations.

        The constants are set as k=2, L=1, and g=9.8.

        Initial set:
            r in [1.19,1.21]
            theta in [0.49,0.51]
            dr in [0,0]
            dtheta in [0,0]

    References
        J. D. Meiss. Differential Dynamical Systems (Monographs on Mathematical Modeling and Computation), Book 14, SIAM publishers, 2007.
    """

    def __init__(self):
        r, theta, dr, dtheta = sp.symbols("r theta dr dtheta")
        xdot = [
             dr,
            dtheta,
            r*dtheta^2 + 9.8*cos(theta) - 2*(r - 1),
            (-2*dr*dtheta - 9.8*sin(theta))/r
        ]
        super(Spring, self).__init__((r, theta, dr, dtheta), xdot)
