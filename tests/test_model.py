# -*- coding: utf-8 -*-

""" Tests for synthetic_control.model. """

from datetime import datetime

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from sklearn.linear_model import Ridge

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
        sc = SyntheticControl(treatment_start=datetime(2009, 1, 1))
        assert isinstance(sc, SyntheticControl)

    def test_setup_model(self):
        """Test the setup_model method."""
        sc = SyntheticControl(
            treatment_start=datetime(2009, 1, 1), treatment_end=datetime(2010, 1, 1)
        )
        sc._setup_model()
        assert isinstance(sc.model, Ridge)
        assert sc.treatment_start == datetime(2009, 1, 1)
        assert sc.treatment_end == datetime(2010, 1, 1)

    def test_create_treatment_phases(self):
        """Test the _create_treatment_phases method."""
        X, y = self.get_test_data()
        sc = SyntheticControl(
            treatment_start=datetime(2009, 1, 1), treatment_end=datetime(2011, 1, 1)
        )
        pre_treatment, treatment, post_treatment = sc._create_treatment_phases(X)
        assert np.allclose(pre_treatment, [True] * 3 + [False] * 4)
        assert np.allclose(treatment, [False] * 3 + [True] * 2 + [False] * 2)
        assert np.allclose(post_treatment, [False] * 5 + [True] * 2)

    def test_create_treatment_phases_no_end(self):
        """Test the _create_treatment_phases method."""
        X, y = self.get_test_data()
        sc = SyntheticControl(treatment_start=datetime(2009, 1, 1))
        pre_treatment, treatment, post_treatment = sc._create_treatment_phases(X)
        assert np.allclose(pre_treatment, [True] * 3 + [False] * 4)
        assert np.allclose(treatment, [False] * 3 + [True] * 4)
        assert np.allclose(post_treatment, [False] * 7)
