# -*- coding: utf-8 -*-

""" Tests for synthetic_control.model. """

from datetime import datetime

from sklearn.linear_model import Ridge

from synthetic_control.model import SyntheticControl


class TestModel:
    def test_create(self):
        sc = SyntheticControl(treatment_start=datetime(2009, 1, 1))
        assert isinstance(sc, SyntheticControl)

    def test_setup_model(self):
        sc = SyntheticControl(
            treatment_start=datetime(2009, 1, 1), treatment_end=datetime(2010, 1, 1)
        )
        sc._setup_model()
        assert isinstance(sc.model, Ridge)
        assert sc.treatment_start == datetime(2009, 1, 1)
        assert sc.treatment_end == datetime(2010, 1, 1)

    def test_create_treatment_phases(self):
        sc = SyntheticControl(treatment_start=datetime(2009, 1, 1))
        model = sc._setup_model(alpha=0.05, positive=False, fit_intercept=True)
        assert isinstance(sc.model, Ridge)
        assert model.alpha == 0.05
        assert model.positive is False
        assert model.fit_intercept is True
