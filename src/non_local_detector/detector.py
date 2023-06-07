import copy
import pickle

import numpy as np
from sklearn.base import BaseEstimator

from non_local_detector.environment import Environment

_DEFAULT_ENVIRONMENT = Environment(environment_name="")


class _DetectorBase(BaseEstimator):
    """Base class for detector objects."""

    def __init__(
        self,
        environments: Environment | list[Environment] = _DEFAULT_ENVIRONMENT,
        infer_track_interior: bool = True,
    ):
        self.environments = environments
        self.infer_track_interior = infer_track_interior

    def fit_environments(
        self, position: np.ndarray, environment_labels: None | np.ndarray = None
    ) -> None:
        """Fits the Environment class on the position data to get information about the spatial environment.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, n_position_dims)
        environment_labels : np.ndarray, optional, shape (n_time,)
            Labels for each time points about which environment it corresponds to, by default None

        """
        for environment in self.environments:
            if environment_labels is None:
                is_environment = np.ones((position.shape[0],), dtype=bool)
            else:
                is_environment = environment_labels == environment.environment_name
            environment.fit_place_grid(
                position[is_environment], infer_track_interior=self.infer_track_interior
            )

    def fit(self):
        """To be implemented by inheriting class"""
        raise NotImplementedError

    def predict(self):
        """To be implemented by inheriting class"""
        raise NotImplementedError

    def fit_predict(self):
        """To be implemented by inheriting class"""
        raise NotImplementedError

    def save_model(self, filename: str = "model.pkl"):
        """Save the detector to a pickled file.

        Parameters
        ----------
        filename : str, optional

        """
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_model(filename: str = "model.pkl"):
        """Load the detector from a file.

        Parameters
        ----------
        filename : str, optional

        Returns
        -------
        detector instance

        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def copy(self):
        """Makes a copy of the detector"""
        return copy.deepcopy(self)


class ClusterlessDetector(_DetectorBase):
    pass


class SortedSpikesDetector(_DetectorBase):
    pass
