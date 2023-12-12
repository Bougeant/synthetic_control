# -*- coding: utf-8 -*-

""" A class to use the synthetic control method. """

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge


class SyntheticControl:
    """A model to use the synthetic control method.

    Parameters
    ----------
    treatment_start : datetime
        The start of the treatment period.
    treatment_end : datetime
        The end of the treatment period.
    **kwargs
        Additional keyword arguments to pass to the model.
    """

    CI_PERCENTILES = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]

    def __init__(
        self,
        treatment_start,
        treatment_end=None,
        ci_sample_size=100,
        ci_fraction=0.1,
        ci_percentiles=CI_PERCENTILES,
        **kwargs,
    ):
        self.treatment_start = treatment_start
        self.treatment_end = treatment_end
        self.ci_sample_size = ci_sample_size
        self.ci_fraction = ci_fraction
        self.ci_percentiles = ci_percentiles
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
        """Generate the predictions for the synthetic control group.

        Parameters
        ----------
        X : pandas.DataFrame
            The data used to build the synthetic control group.

        Returns
        -------
        y_pred : pandas.Series
            The predictions for the synthetic control group.
        """
        return self.model.predict(X)

    def fit_predict(self, X, y):
        """Fit the model to the data outside of the treatment period and generate
        the predictions for the synthetic control group.

        Parameters
        ----------
        X : pandas.DataFrame
            The data used to build the synthetic control group.
        y : pandas.Series
            The treatment group to match outside of the treatment period.

        Returns
        -------
        y_pred : pandas.Series
            The predictions for the synthetic control group.
        """
        self.fit(X, y)
        return self.model.predict(X)

    def get_confidence_interval(self, X, y):
        """Return the confidence interval on the predictions for the synthetic
        control group.

        Parameters
        ----------
        X : pandas.DataFrame
            The data used to build the synthetic control group.
        y : pandas.Series
            The treatment group to match outside of the treatment period.

        Returns
        -------
        y_pred : pandas.DataFrame
            The quantile predictions for the synthetic control group.
        """
        preds = []
        for _ in range(self.ci_sample_size):
            X_iter = X.T.sample(frac=self.ci_fraction).T
            model = clone(self.model)
            model.fit(X_iter, y)
            y_pred = model.predict(X_iter)
            preds.append(y_pred)
        ci = {q: np.percentile(preds, q=q, axis=0) for q in self.ci_percentiles}
        return pd.DataFrame(ci, index=X.index)
