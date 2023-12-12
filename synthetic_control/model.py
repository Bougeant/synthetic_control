# -*- coding: utf-8 -*-

""" A class to use the synthetic control method. """

from sklearn.linear_model import Ridge


class SyntheticControl:
    """A model to use the synthetic control method."""

    def __init__(self, treatment_start, treatment_end=None, **kwargs):
        self.treatment_start = treatment_start
        self.treatment_end = treatment_end
        self.model = self._setup_model(**kwargs)

    def _setup_model(self, alpha=1.0, positive=True, fit_intercept=False):
        """Setup the model to build the synthetic control group.

        Parameters
        ----------
        alpha : float
            L2 regularization strength.
        positive : bool
            Force the linear coefficients to be positive. This is recommended for
            creating a reasonable synthetic control group.
        fit_intercept : bool
            Whether to fit the intercept for this model. It is recommended to se
            if to False for creating a reasonable synthetic control group.

        Returns
        -------
        sklearn.linear_model.Ridge
            A Ridge regression model used to build the synthetic control group.
        """
        return Ridge(alpha=alpha, positive=positive, fit_intercept=fit_intercept)

    def _get_treatment_phase(self, X):
        """Return the treatment phase. During the treatment itself, the model
        will not attempt to fit the synthetic control group to the treatment group.

        Parameters
        ----------
        X : pandas.DataFrame
            The data used to build the synthetic control group.

        Returns
        -------
        treatment : pandas.Series
            A boolean series indicating whether the observation occurs during the
            treatment.
        """
        treatment = X.index >= self.treatment_start
        if self.treatment_end:
            treatment = treatment & (X.index < self.treatment_end)
        return treatment

    def fit(self, X, y):
        """Fit the model to the data outside of the treatment period.

        Parameters
        ----------
        X : pandas.DataFrame
            The data used to build the synthetic control group.
        y : pandas.Series
            The treatment group to match outside of the treatment period.

        Returns
        -------
        self : SyntheticControl
            The SyntheticControl with the fitted model.
        """
        fitting_period = ~self._get_treatment_phase(X)
        X_train = X.loc[fitting_period]
        y_train = y.loc[fitting_period]
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.model.predict(X)
