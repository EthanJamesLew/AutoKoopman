from autokoopman.core.estimator import TrajectoryEstimator
from typing import Sequence
import random
import abc
from autokoopman.core.format import _clip_list


class Parameter:
    def __init__(self, name):
        self._name = name

    @abc.abstractmethod
    def random(self):
        pass

    @abc.abstractmethod
    def is_member(self, item) -> bool:
        ...

    def __contains__(self, item) -> bool:
        return self.is_member(item)

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f"<{self.__class__.__name__} Name: {self.name}>"


class FiniteParameter(Parameter):
    def __init__(self, name: str, elements: Sequence):
        super(FiniteParameter, self).__init__(name)
        self._elements = tuple(elements)

    def is_member(self, item) -> bool:
        return item in self._elements

    def random(self):
        return random.choice(self._elements)


class ContinuousParameter(Parameter):
    def __init__(self, name: str, domain_lower, domain_upper):
        super(ContinuousParameter, self).__init__(name)
        assert domain_upper >= domain_lower
        self._interval = (domain_lower, domain_upper)

    def is_member(self, item) -> bool:
        return item >= self._interval[0] and item <= self._interval[1]

    def random(self):
        return random.random() * (self._interval[1] - self._interval[0]) + self._interval[0]


class DiscreteParameter(FiniteParameter):
    def __init__(self, name: str, domain_lower: int, domain_upper: int, step=1):
        super(DiscreteParameter, self).__init__(name, range(domain_lower, domain_upper, step))


class ParameterSpace(Parameter):
    def __init__(self, name: str, coords: Sequence[Parameter]):
        super(ParameterSpace, self).__init__(name)
        self._coords = coords

    def is_member(self, item) -> bool:
        return all([itemi in coordi for itemi, coordi in zip(item, self._coords)])

    def random(self):
        return [coordi.random() for coordi in self._coords]

    @property
    def dimension(self):
        return len(self._coords)

    def __repr__(self):
        return f"<{self.__class__.__name__} Name: {self.name} Dimensions: {self.dimension} " \
               f"Coordinates: {_clip_list([s.name+f': {s.__class__.__name__}' for s in self._coords])}>"


class HyperparameterMap:
    """
    define and associate a hyperparameter space with a moddel
    """
    def __init__(self, parameter_space: ParameterSpace):
        self.parameter_space = parameter_space

    @abc.abstractmethod
    def get_model(hyperparams: Sequence) -> TrajectoryEstimator:
        raise NotImplementedError


class HyperparameterTuner:
    """
    with training data and a model, determine ideal hyperparameters
    """
    ...
