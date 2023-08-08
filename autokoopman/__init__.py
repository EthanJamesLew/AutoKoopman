# TODO: update this
__author__ = "Ethan Lew"
__copyright__ = "Copyright 2023"
__credits__ = ["Ethan Lew", "Abdelrahman Hekal", "Kostiantyn Potomkin", "Niklas Kochdumper", "Brandon Hencey", "Stanley Bak", "Sergiy Bogomolov"]
__license__ = "GPLv3"
__version__ = "0.21"
__maintainer__ = "Ethan Lew"
__email__ = "ethanlew16@gmail.com"
__status__ = "Prototype"

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
