# -*- coding: utf-8 -*-

""" A class to use the synthetic control method. """

from sklearn.linear_model import Ridge


class SyntheticControl:
    """A model to use the synthetic control method."""

    def __init__(self, force_positive=True, fit_intercept=False):
        self.force_positive = force_positive
        self.fit_intercept = fit_intercept
        self.model = self.setup_model()

    def setup_model(self):
        self.model = Ridge(
            fit_intercept=self.fit_intercept, positive=self.force_positive
        )

    def fit(self):
        pass

    def predict(self):
        pass

    def fit_predict(self):
        pass
