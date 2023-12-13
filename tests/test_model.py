# -*- coding: utf-8 -*-

""" Tests for synthetic_control.model. """

from datetime import datetime

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted

from synthetic_control.model import SyntheticControl


class TestSyntheticControl:
    """Test the SyntheticControl class."""

    def get_test_data(self):
        """Create a test dataset."""
        index = pd.date_range(start="2006-01-01", end="2012-01-01", freq="YS")
        X = pd.DataFrame(
            {
                "Cat": [80, 84, 78, 90, 76, 72, 80],
                "Dog": [20, 16, 22, 10, 24, 28, 20],
                "Bird": [120, 80, 75, 182, 80, 60, 80],
                "Fish": [200, 400, 80, 129, 280, 320, 200],
            },
            index=index,
        )
        y = pd.Series(data=[100, 97, 96, 104, 92, 86, 92], index=index)
        return X, y

    def test_create(self):
        """Test the creation of a SyntheticControl object."""
        sc = SyntheticControl(treatment_start=datetime(2008, 1, 1))
        assert isinstance(sc, SyntheticControl)

    def test_setup_model(self):
        """Test the setup_model method."""
        sc = SyntheticControl(
            treatment_start=datetime(2008, 1, 1), treatment_end=datetime(2010, 1, 1)
        )
        sc._setup_model()
        assert isinstance(sc.model, Ridge)
        assert sc.treatment_start == datetime(2008, 1, 1)
        assert sc.treatment_end == datetime(2010, 1, 1)

    def test_get_treatment_phase(self):
        """Test the _create_treatment_phases method."""
        X, y = self.get_test_data()
        sc = SyntheticControl(
            treatment_start=datetime(2008, 1, 1), treatment_end=datetime(2010, 1, 1)
        )
        treatment = sc._get_treatment_phase(X)
        assert np.allclose(treatment, [False] * 3 + [True] * 2 + [False] * 2)

    def test_get_treatment_phase_no_end(self):
        """Test the _create_treatment_phases method."""
        X, y = self.get_test_data()
        sc = SyntheticControl(treatment_start=datetime(2008, 1, 1))
        treatment = sc._get_treatment_phase(X)
        assert np.allclose(treatment, [False] * 3 + [True] * 4)

    def test_fit(self):
        """Test the fit method."""
        X, y = self.get_test_data()
        sc = SyntheticControl(treatment_start=datetime(2008, 1, 1))
        sc.fit(X, y)
        check_is_fitted(sc.model)

    def test_predict(self):
        """Test the predict method."""
        X, y = self.get_test_data()
        sc = SyntheticControl(treatment_start=datetime(2008, 1, 1))
        sc.fit(X, y)
        y_synth = sc.predict(X)
        np.allclose(y_synth, [100.0, 97.1, 95.91, 106.6, 96.1, 93.9, 96.6], atol=0.1)

    def test_fit_predict(self):
        """Test the fit_predict method."""
        X, y = self.get_test_data()
        sc = SyntheticControl(treatment_start=datetime(2008, 1, 1))
        y_synth = sc.fit_predict(X, y)
        np.allclose(y_synth, [100.0, 97.1, 95.91, 106.6, 96.1, 93.9, 96.6], atol=0.1)

    def test_get_confidence_interval(self):
        """Test the get_confidence_interval method."""
        X, y = self.get_test_data()
        sc = SyntheticControl(
            treatment_start=datetime(2008, 1, 1),
            ci_fraction=0.6,
            ci_percentiles=[5, 95],
        )
        y_ci = sc.get_confidence_interval(X, y)
        expected_df = pd.DataFrame(
            {
                5: [96.8, 81.2, 71.4, 51.4, 86.5, 70.7, 82.3],
                95: [118.1, 101.6, 100.1, 169.9, 121.1, 140.7, 98.2],
            },
            index=X.index,
        )
        assert_frame_equal(y_ci, expected_df, atol=0.1)
