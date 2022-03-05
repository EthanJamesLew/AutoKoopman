from typing import Optional, Tuple, Sequence

import numpy as np
import pysindy as ps  # type: ignore

import autokoopman.core.estimator as aest
import autokoopman.core.system as asys
import autokoopman.core.trajectory as atraj
import autokoopman.core.tuner as atune


class FourierSindy(atune.HyperparameterMap):
    """SINDy with hyperparameters for Fourier library"""

    def __init__(self, names: Sequence[str], threshold_range: Tuple[float, float]):
        self.names = names
        super(FourierSindy, self).__init__(
            atune.ParameterSpace(
                "sindy-fourier",
                [
                    atune.ContinuousParameter(
                        "sr3-threshold", *threshold_range, distribution="loguniform"
                    ),
                    atune.DiscreteParameter("n-frequency", 1, 4),
                ],
            )
        )

    def get_model(self, hyperparams: Sequence) -> aest.TrajectoryEstimator:
        model = ps.SINDy(
            feature_names=self.names,
            differentiation_method=ps.FiniteDifference(),
            optimizer=ps.SR3(threshold=hyperparams[0], thresholder="l1"),
            feature_library=ps.FourierLibrary(n_frequencies=hyperparams[1]),
        )
        return SindyEstimator(model)


class PolynomialSindy(atune.HyperparameterMap):
    """SINDy with hyperparameters for polynomial library"""

    def __init__(self, names: Sequence[str], threshold_range: Tuple[float, float]):
        self.names = names
        super(PolynomialSindy, self).__init__(
            atune.ParameterSpace(
                "sindy-poly",
                [
                    atune.ContinuousParameter(
                        "sr3-threshold", *threshold_range, distribution="loguniform"
                    ),
                    atune.DiscreteParameter("poly-degree", 1, 4),
                ],
            )
        )

    def get_model(self, hyperparams: Sequence) -> aest.TrajectoryEstimator:
        model = ps.SINDy(
            feature_names=self.names,
            differentiation_method=ps.FiniteDifference(),
            optimizer=ps.SR3(threshold=hyperparams[0], thresholder="l1"),
            feature_library=ps.PolynomialLibrary(degree=hyperparams[1]),
        )
        return SindyEstimator(model)


class SindyEstimator(aest.TrajectoryEstimator):
    def __init__(self, sindy: Optional[ps.SINDy] = None):
        self._usr_sindy = sindy

    def fit(self, X: atraj.TrajectoriesData) -> None:
        if self._usr_sindy is None:
            self._model = ps.SINDy(
                feature_names=X.state_names,
                differentiation_method=ps.FiniteDifference(),
                optimizer=ps.SR3(threshold=0.04, thresholder="l1"),
                feature_library=ps.PolynomialLibrary(degree=3),
            )

        else:
            self._model = self._usr_sindy
        self._model.fit(
            [xi.states for xi in X],
            t=[xi.times for xi in X],
            multiple_trajectories=True,
        )

    def predict(self, iv: np.ndarray, times: np.ndarray) -> atraj.Trajectory:
        ret = self._model.simulate(iv, times)
        return atraj.Trajectory(times, ret, self._model.feature_names)

    @property
    def model(self) -> asys.ContinuousSystem:
        def gradient_f(t, x):
            return self._model.predict(np.atleast_2d(x))

        return asys.GradientContinuousSystem(gradient_f, self._model.feature_names)
