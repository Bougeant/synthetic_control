# -*- coding: utf-8 -*-

""" A class to use the synthetic control method. """

from sklearn.linear_model import Ridge


class SyntheticControl:
    """A model to use the synthetic control method."""

    def __init__(self):
        self.model = Ridge(fit_intercept=False, positive=True)

    def fit(self):
        pass

    def predict(self):
        pass

    def fit_predict(self):
        pass
