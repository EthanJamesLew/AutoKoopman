"""Pipelines 
    map parameters spaces to execution artifacts
"""
import copy
import abc
from typing import Optional, Sequence, Any
from autokoopman.core.hyperparameter import ParameterSpace
from autokoopman.core.trajectory import TrajectoriesData


class Pipeline:
    this_parameter_space = ParameterSpace("", [])

    def __init__(self, name: str) -> None:
        self._param_space = self.this_parameter_space
        self._next_stages: Sequence[Pipeline] = []
        self.name = name

    @abc.abstractmethod
    def execute(self, inputs, params: Optional[Sequence[Any]]) -> Any:
        """execute this stage only"""
        raise NotImplementedError

    def run(self, inputs, params: Optional[Sequence[Any]]):
        """run full pipeline"""
        # input checks
        assert params in self.parameter_space
        params = self._split_inputs(params)

        # run the current stage
        results = self.execute(inputs, params[0])

        # if no other stages, return the results, else flow them through the following stages
        if len(self._next_stages) == 0:
            return results
        else:
            rem = [
                stage.run(results, p) for stage, p in zip(self._next_stages, params[1:])
            ]
            if len(rem) == 1:
                return rem[0]
            else:
                return tuple(rem)

    def add_post_pipeline(self, next_pipeline: "Pipeline"):
        assert isinstance(
            next_pipeline, Pipeline
        ), f"next pipeline must be a Pipeline object"
        self._next_stages.append(next_pipeline)

    def __or__(self, next: Any):
        if not isinstance(next, Pipeline):
            raise ValueError(f"{next} must be a Pipeline")

        # create a new instance
        n = copy.deepcopy(self)
        n.add_post_pipeline(next)
        return n

    def _split_inputs(self, inputs: Sequence[Any]):
        idx = self._param_space.dimension
        inps = [inputs[0:idx]]
        for stage in self._next_stages:
            inps.append(inputs[idx : (idx + stage.parameter_space.dimension)])
            idx += stage.parameter_space.dimension
        return inps

    @property
    def parameter_space(self):
        return ParameterSpace.from_parameter_spaces(
            self.name,
            [
                self._param_space,
                *[stage.parameter_space for stage in self._next_stages],
            ],
        )


class TrajectoryPreprocessor(Pipeline):
    def run(self, inputs, params: Optional[Sequence[Any]]) -> TrajectoriesData:
        return super().run(inputs, params)
