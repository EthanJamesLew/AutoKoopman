"""
Model has:

* a set of hyperparameters
"""
import abc
from typing import Optional

from sklearn.preprocessing import MinMaxScaler

import numpy as np

import autokoopman.core.system as asys
import autokoopman.core.trajectory as atraj


class TrajectoryEstimator(abc.ABC):
    """
    Trajectory based estimator base class

    :param normalize: apply MinMax normalization to the fit data
    :param feature_range: range for MinMax scaler
    """

    def __init__(self, normalize=True, feature_range=(-1, 1)) -> None:
        self.normalize = normalize
        if self.normalize:
            self.scaler = MinMaxScaler(feature_range=feature_range)
        else:
            self.scaler = None

    @abc.abstractmethod
    def fit(self, X: atraj.TrajectoriesData) -> None:
        pass

    @property
    @abc.abstractmethod
    def model(self) -> asys.System:
        pass


class NextStepEstimator(TrajectoryEstimator):
    """Estimator of discrete time dynamical systems

    Requires that the data be uniform time

    :param normalize: apply MinMax normalization to the fit data
    :param feature_range: range for MinMax scaler
    """

    def __init__(self, normalize=True, feature_range=(-1, 1), weights=None) -> None:
        super().__init__(normalize=normalize, feature_range=feature_range)
        self.names = None
        self.weights = None

    @abc.abstractmethod
    def fit_next_step(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        U: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        """an alternative fit method that uses a trajectories data structure"""
        pass

    def fit(self, X: atraj.TrajectoriesData) -> None:
        assert isinstance(
            X, atraj.UniformTimeTrajectoriesData
        ), "X must be uniform time"
        if self.weights is None:
            X_, Xp_, U_ = X.next_step_matrices
            W = None
        else:
            X_, Xp_, U_, W = X.next_step_matrices_weights(self.weights)
        if self.normalize:
            X_ = self.scaler.fit_transform(X_.T).T
            Xp_ = self.scaler.transform(Xp_.T).T
        self.fit_next_step(X_, Xp_, U_, W)
        self.sampling_period = X.sampling_period
        self.names = X.state_names


class GradientEstimator(TrajectoryEstimator):
    """Estimator of discrete time dynamical systems

    Requires that the data be uniform time

    :param normalize: apply MinMax normalization to the fit data
    :param feature_range: range for MinMax scaler
    """

    def __init__(self, normalize=True, feature_range=(-1, 1), weights=None) -> None:
        super().__init__(normalize=normalize, feature_range=feature_range)
        self.names = None
        self.weights = None

    @abc.abstractmethod
    def fit_gradient(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        U: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        """an alternative fit method that uses a trajectories data structure"""
        pass

    def fit(self, X: atraj.TrajectoriesData) -> None:
        assert isinstance(
            X, atraj.UniformTimeTrajectoriesData
        ), "X must be uniform time"
        if self.weights is None:
            X_, Xp_, U_ = X.differentiate
            W = None
        else:
            X_, Xp_, U_, W = X.differentiate_weights(self.weights)
        if self.normalize:
            X_ = self.scaler.fit_transform(X_.T).T
            Xp_ = self.scaler.transform(Xp_.T).T
        self.fit_gradient(X_, Xp_, U_, W)
        self.sampling_period = X.sampling_period
        self.names = X.state_names
