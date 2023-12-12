# -*- coding: utf-8 -*-

""" A class to use the synthetic control method. """

from sklearn.linear_model import Ridge


class SyntheticControl:
    """A model to use the synthetic control method."""

    def __init__(
        self,
        treatment_start,
        treatment_end=None,
        force_positive=True,
        fit_intercept=False,
    ):
        self.treatment_start = treatment_start
        self.treatment_end = treatment_end
        self.model = self.setup_model(force_positive, fit_intercept)

    def setup_model(self, force_positive, fit_intercept):
        return Ridge(
            positive=force_positive,
            fit_intercept=fit_intercept,
        )

    def fit(self):
        pass

    def predict(self):
        pass

    def fit_predict(self):
        pass
