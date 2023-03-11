"""
AutoKoopman: Automated Koopman operator methods for data-driven dynamical 
systems analysis and control.

The AutoKoopman package provides a high-level system identification tool 
that automatically optimizes all hyper-parameters to estimate accurate system 
models with globally linearized representations. Implemented as a Python library 
under shared class interfaces, AutoKoopman uses a collection of Koopman-based 
algorithms centered on conventional dynamic mode decomposition and deep learning.

The package includes functions for generating Koopman operator matrices from 
time series data, as well as tools for computing eigenvalues and eigenvectors 
of these matrices. AutoKoopman supports both discrete-time and continuous-time 
system identification, and includes methods for extended dynamic mode 
decomposition (EDMD), deep Koopman, and sparse identification of nonlinear 
dynamics (SINDy). The package also provides major types of static observables, 
including polynomial observables, neural network observables, and random 
Fourier feature observables.

AutoKoopman supports system identification with input and control, including 
the Koopman operator with input and control (KIC) method. The package also 
includes methods for online (streaming) system identification, including 
online dynamic mode decomposition (DMD).

Finally, AutoKoopman includes hyperparameter optimization methods for tuning 
model parameters, including random search, grid search, and Bayesian 
optimization.

Use Cases
----------
The library is intended for a systems engineer/researcher who wishes to 
leverage data-driven dynamical systems techniques. The user may have 
measurements of their system with no prior model.

Prediction: AutoKoopman can predict the evolution of a system over long time
horizons by leveraging the globally linearized representation provided by 
Koopman operator methods.

Control: AutoKoopman can synthesize control signals that achieve desired 
closed-loop behaviors and are optimal with respect to some objective by 
using the Koopman operator with input and control (KIC) method.

Verification: AutoKoopman can prove or falsify the safety requirements of 
a system by providing a mathematical representation of the system dynamics 
that can be analyzed and verified.

For more information on the theory behind Koopman operator methods and 
their use in system identification, see the following references:

- Koopman, B. O. (1931). Hamiltonian systems and transformation in Hilbert 
space. Proceedings of the National Academy of Sciences, 17(5), 315-318.
- Mezic, I. (2013). Analysis of fluid flows via spectral properties of the 
Koopman operator. Annual Review of Fluid Mechanics, 45, 357-378.
- Proctor, J. L., & Brunton, S. L. (2020). Koopman operator-based system 
identification: a survey of methods and recent advances. arXiv preprint 
arXiv:2009.14544.

For usage examples and additional documentation, see the package 
documentation.
"""

__author__ = "Ethan Lew"
__copyright__ = "Copyright 2023"
__credits__ = [
    "Ethan Lew",
    "Abdelrahman Hekal",
    "Kostiantyn Potomkin",
    "Niklas Kochdumper",
    "Brandon Hencey",
    "Stanley Bak",
    "Sergiy Bogomolov",
]
__license__ = "GPLv3"
__maintainer__ = "Ethan Lew"
__email__ = "ethanlew16@gmail.com"
__status__ = "Prototype"

# we auto-manage versions
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from autokoopman.autokoopman import auto_koopman

from autokoopman.core.system import (
    ContinuousSystem,
    DiscreteSystem,
    GradientContinuousSystem,
    SymbolicContinuousSystem,
)
from autokoopman.core.trajectory import (
    TrajectoriesData,
    Trajectory,
    UniformTimeTrajectoriesData,
    UniformTimeTrajectory,
)

from . import _version

__version__ = _version.get_versions()["version"]
