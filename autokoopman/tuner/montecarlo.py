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
        next(sampling)

        while True:
            param = self._parameter_model.parameter_space.random()
            try:
                sampling.send(param)
                next(sampling)
            except StopIteration:
                break
            except Exception as exc:
                print(f"Error: {exc}")
                self.error_messages.append((param, exc))
        return self.best_result
