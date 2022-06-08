import sympy as sp  # type: ignore
from numpy import cos, sin

import autokoopman.core.system as asys


class Spring(asys.SymbolicContinuousSystem):
    r"""
    Spring pendulum
        We study the behavior of the planar spring-pendulum described in [1].
        It consists of a solid ball of mass :math:`m` and a spring of natural length :math:`L`.
        The spring constant is :math:`k`.

        We study the evolutions of the length r of the spring and the angle :math:`\theta`
        between the spring and the vertical. They are modeled by the following differential
        equations.

        The constants are set as :math:`k=2`, :math:`L=1`, and :math:`g=9.8`.

        .. math::
            \begin{bmatrix} \dot r \\ \dot \theta \\ \dot d_r \\ \dot d_{\theta} \end{bmatrix} = \begin{bmatrix} d_r \\ d_{\theta} \\ r d_{\theta}^2 + g * \cos (\theta) -2 (r - 1) \\ \left( -2 d_r d_{\theta} - g \sin (\theta) \right) / 2 \end{bmatrix}

        Initial set:
            r in [1.19,1.21]
            theta in [0.49,0.51]
            dr in [0,0]
            dtheta in [0,0]

    References
        J. D. Meiss. Differential Dynamical Systems (Monographs on Mathematical Modeling and Computation), Book 14, SIAM publishers, 2007.
    """

    def __init__(self, g=9.81):
        r, theta, dr, dtheta = sp.symbols("r theta dr dtheta")
        xdot = [
            dr,
            dtheta,
            r * dtheta**2 + g * cos(theta) - 2 * (r - 1),
            (-2 * dr * dtheta - g * sin(theta)) / r,
        ]
        super(Spring, self).__init__((r, theta, dr, dtheta), xdot)
