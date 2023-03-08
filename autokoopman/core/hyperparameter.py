"""Mini-Language to Express Hyperparameter Sets

@TODO: look into backends to make this more robust
"""
import abc
from typing import Sequence, Callable
import random

import numpy as np

from autokoopman.core.format import _clip_list


class Parameter:
    """hyperparameter is a set that you can
        * name
        * sample randomly
        * check membership

    @param name: parameter identifier
    """

    def __init__(self, name):
        self._name = name

    @abc.abstractmethod
    def random(self):
        """get an element from the parameter at random"""
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
    """a finite set of things"""

    def __init__(self, name: str, elements: Sequence):
        super(FiniteParameter, self).__init__(name)
        self.elements = tuple(elements)

    def is_member(self, item) -> bool:
        return item in self.elements

    def random(self):
        return random.choice(self.elements)


class ContinuousParameter(Parameter):
    """a continuous, closed interval"""

    @staticmethod
    def loguniform(low=0.1, high=1, size=None):
        return np.exp(np.random.uniform(np.log(low), np.log(high), size))

    @staticmethod
    def uniform(low=0, high=1, size=None):
        return np.random.uniform(low, high, size)

    def __init__(self, name: str, domain_lower, domain_upper, distribution="uniform"):
        super(ContinuousParameter, self).__init__(name)
        assert domain_upper >= domain_lower
        self._interval = (domain_lower, domain_upper)
        self.distribution = distribution

    def is_member(self, item) -> bool:
        return item >= self._interval[0] and item <= self._interval[1]

    def random(self):
        if isinstance(self.distribution, Callable):
            return self.distribution()
        elif hasattr(self, self.distribution):
            return getattr(self, self.distribution)(
                self._interval[0], self._interval[1]
            )
        else:
            raise ValueError(f"cannot find distribution {self.distribution}")


class DiscreteParameter(FiniteParameter):
    """a range object"""

    def __init__(self, name: str, domain_lower: int, domain_upper: int, step=1):
        super(DiscreteParameter, self).__init__(
            name, range(domain_lower, domain_upper, step)
        )


class ParameterSpace(Parameter):
    """an interval hull"""

    def __init__(self, name: str, coords: Sequence[Parameter]):
        super(ParameterSpace, self).__init__(name)
        self._coords = coords
        self._cdict = {c.name: c for c in self._coords}

    def is_member(self, item) -> bool:
        return all([itemi in coordi for itemi, coordi in zip(item, self._coords)])

    def random(self):
        return [coordi.random() for coordi in self._coords]

    def __getitem__(self, item):
        assert (
            item in self._cdict
        ), f"coordinate {item} was not found in space (values are {list(self._cdict.keys())})"
        return self._cdict[item]

    def __iter__(self):
        for c in self._coords:
            yield c

    @property
    def dimension(self):
        return len(self._coords)

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} Name: {self.name} Dimensions: {self.dimension} "
            f"Coordinates: {_clip_list([s.name+f': {s.__class__.__name__}' for s in self._coords])}>"
        )
