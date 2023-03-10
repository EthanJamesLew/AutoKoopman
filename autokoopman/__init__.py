# TODO: update this
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
