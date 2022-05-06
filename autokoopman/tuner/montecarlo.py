import autokoopman.core.tuner as atuner
from autokoopman.core.trajectory import TrajectoriesData
from autokoopman.core.tuner import TuneResults, TrajectoryScoring
from typing import Callable


class MonteCarloTuner(atuner.HyperparameterTuner):
    def tune(
        self,
        nattempts=100,
        scoring_func: Callable[
            [TrajectoriesData, TrajectoriesData], float
        ] = TrajectoryScoring.end_point_score,
    ) -> TuneResults:
        sampling = self.tune_sampling(nattempts, scoring_func)
        sampling.send(None)
        while True:
            try:
                param = self._parameter_model.parameter_space.random()
                sampling.send(param)
            except StopIteration:
                break
        return self.best_result
