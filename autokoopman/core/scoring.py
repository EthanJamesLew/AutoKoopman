"""Scoring Metrics for Evaluation
"""
import numpy as np

from typing import Dict, Hashable
from autokoopman.core.trajectory import TrajectoriesData


class TrajectoryScoring:
    @staticmethod
    def weighted_score(
        true_data: TrajectoriesData,
        prediction_data: TrajectoriesData,
        weights: Dict[Hashable, np.ndarray],
    ):
        raise NotImplementedError("weighted scoring not yet implemented")

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
