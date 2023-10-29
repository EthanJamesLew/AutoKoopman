import abc
import random
from typing import Sequence, Callable, TypedDict, Any
import numpy as np

from autokoopman.core.estimator import TrajectoryEstimator
from autokoopman.core.trajectory import (
    TrajectoriesData,
    UniformTimeTrajectory,
    UniformTimeTrajectoriesData,
)
from autokoopman.core.format import _clip_list
from sklearn.model_selection import KFold


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
        self.elements = tuple(elements)

    def is_member(self, item) -> bool:
        return item in self.elements

    def random(self):
        return random.choice(self.elements)


class ContinuousParameter(Parameter):
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
    def __init__(self, name: str, domain_lower: int, domain_upper: int, step=1):
        super(DiscreteParameter, self).__init__(
            name, range(domain_lower, domain_upper, step)
        )


class ParameterSpace(Parameter):
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


class HyperparameterMap:
    """
    define and associate a hyperparameter space with a moddel
    """

    def __init__(self, parameter_space: ParameterSpace):
        self.parameter_space = parameter_space

    @abc.abstractmethod
    def get_model(self, hyperparams: Sequence) -> TrajectoryEstimator:
        raise NotImplementedError


class TuneResults(TypedDict):
    model: TrajectoryEstimator
    param: Sequence[Any]
    score: float


class TrajectoryScoring:
    @staticmethod
    def end_point_score(true_data: TrajectoriesData, prediction_data: TrajectoriesData):
        errors = (prediction_data - true_data).norm()
        end_errors = np.array([s.states[-1] for s in errors])
        return np.mean(end_errors)

    @staticmethod
    def total_score(true_data: TrajectoriesData, prediction_data: TrajectoriesData):
        errors = (prediction_data - true_data).norm()
        end_errors = np.array([s.states.flatten() for s in errors])

        return np.mean(np.concatenate(end_errors, axis=0))

    @staticmethod
    def relative_score(true_data: TrajectoriesData, prediction_data: TrajectoriesData):
        # TODO: implement this
        err_term = []
        for k in prediction_data.traj_names:
            pred_t = prediction_data[k]
            true_t = true_data[k]
            abs_error = np.linalg.norm(pred_t.states - true_t.states)
            mean_error = np.linalg.norm(pred_t.states - np.mean(pred_t.states, axis=0))
            err_term.append(abs_error / mean_error)
        return np.mean(np.array(err_term))


class HyperparameterTuner(abc.ABC):
    """
    Tune a HyperparameterMap Object
        The tuner implements a model learning pipeline that can take a mapping from hyperparameters to a model and
        training data and learn an optimized model. For example, given a SINDy estimator, hyperparameter tuner can
        implement gridsearch to find the best sparsity threshold and library type.
    """

    @staticmethod
    def generate_predictions(
        trained_model: TrajectoryEstimator, holdout_data: TrajectoriesData
    ):
        preds = {}
        # get the predictions
        for k, v in holdout_data._trajs.items():
            sivp_interp = trained_model.model.solve_ivp(
                v.states[0],
                (np.min(v.times), np.max(v.times)),
                inputs=v.inputs,
                teval=v.times,
            )
            # sivp_interp = sivp_interp.interp1d(v.times)
            preds[k] = sivp_interp
        return TrajectoriesData(preds)

    def __init__(
        self,
        parameter_model: HyperparameterMap,
        training_data: TrajectoriesData,
        n_splits=None,
        display_progress: bool = True,
        verbose: bool = True,
    ):
        self._parameter_model = parameter_model
        self._training_data = training_data
        self.scores = []
        self.best_scores = []
        self.best_result = None
        self.error_messages = []
        self.n_splits = n_splits
        self.verbose = verbose
        self.disp_progress = display_progress

    def _reset_scores(self):
        self.scores = []
        self.best_scores = []
        self.best_result = None

    @abc.abstractmethod
    def tune(
        self,
        nattempts=100,
        scoring_func: Callable[
            [TrajectoriesData, TrajectoriesData], float
        ] = TrajectoryScoring.total_score,
    ) -> TuneResults:
        pass

    def tune_sampling(
        self,
        nattempts,
        scoring_func: Callable[[TrajectoriesData, TrajectoriesData], float],
    ):
        import tqdm

        self._reset_scores()
        best_model = None
        best_params = None
        best_score = None
        score = None

        if self.n_splits is not None:
            kf = KFold(n_splits=self.n_splits)

        for _ in (
            tqdm.tqdm(
                range(nattempts),
                total=nattempts,
                desc=f"Tuning {self.__class__.__name__}",
            )
            if (self.verbose and self.disp_progress)
            else range(nattempts)
        ):
            param = yield

            assert isinstance(param, Sequence), "yielded param must be a sequence"
            model = self._parameter_model.get_model(param)

            # have something, even if it can't be optimized
            if len(self.scores) == 0:
                self.best_result = TuneResults(
                    model=model,
                    param=param,
                    score=np.infty,
                )

            if self.n_splits is None:
                model.fit(self._training_data)
                prediction_data = self.generate_predictions(model, self._training_data)
                score = scoring_func(self._training_data, prediction_data)
            else:
                iscores = []
                names = np.array(list(self._training_data.traj_names))[:, np.newaxis]
                for train_index, test_index in kf.split(names):
                    # if all trajectories are uniform time, make the data uniform time
                    if all(
                        [
                            isinstance(tinst, UniformTimeTrajectory)
                            for tinst in self._training_data
                        ]
                    ):
                        TrajsType = UniformTimeTrajectoriesData
                    else:
                        TrajsType = TrajectoriesData

                    # create test train split
                    training = TrajsType(
                        {ti[0]: self._training_data[ti[0]] for ti in names[train_index]}
                    )
                    validation = TrajsType(
                        {ti[0]: self._training_data[ti[0]] for ti in names[test_index]}
                    )

                    # do the fit and scoring (select the best median)
                    model = self._parameter_model.get_model(param)
                    model.fit(training)
                    prediction_data = self.generate_predictions(model, validation)
                    iscore = scoring_func(validation, prediction_data)
                    iscores.append(iscore)
                score = np.median(iscores)

            if len(self.scores) == 0 or score < np.min(self.scores):
                best_model = model
                best_params = param
                best_score = score
                assert (
                    (best_model is not None)
                    and (best_params is not None)
                    and (best_score is not None)
                )
                self.best_result = TuneResults(
                    model=best_model, param=best_params, score=best_score
                )
            self.scores.append(score if not np.isnan(score) else 1e12)
            self.best_scores.append(np.min(self.scores))
            yield score
