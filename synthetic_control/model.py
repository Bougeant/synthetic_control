# -*- coding: utf-8 -*-

""" A class to use the synthetic control method. """

from sklearn.linear_model import Ridge


class SyntheticControl:
    """A model to use the synthetic control method."""

    def __init__(
        self,
        treatment_start,
        treatment_end=None,
        **kwargs,
    ):
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
        model : sklearn.linear_model.Ridge
            A Ridge regression model used to build the synthetic control group.
        """
        return Ridge(alpha=alpha, positive=positive, fit_intercept=fit_intercept)

    def _create_treatment_phases(self, X):
        """Create the treatment phases. During the treatment itself, the model
        will not attempt to fit the synthetic control group to the treatment group.

        Parameters
        ----------
        X : pandas.DataFrame
            The data used to build the synthetic control group.

        Returns
        -------
        pre_treatment : pandas.Series
            A boolean series indicating whether the observation is before the
            treatment.
        treatment : pandas.Series
            A boolean series indicating whether the observation is during the
            treatment.
        post_treatment : pandas.Series
            A boolean series indicating whether the observation is after the
            treatment.
        """
        pre_treatment = X.index < self.treatment_start
        if self.treatment_end:
            post_treatment = X.index >= self.treatment_end
        else:
            post_treatment = False
        treatment = ~pre_treatment & ~post_treatment
        return pre_treatment, treatment, post_treatment

    def fit(self, X, y):
        pass

    def predict(self):
        pass

    def fit_predict(self):
        pass
