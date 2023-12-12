# -*- coding: utf-8 -*-

""" Tests for synthetic_control.model. """

from sklearn.linear_model import Ridge

from synthetic_control.model import SyntheticControl


class TestModel:
    def test_create(self):
        sc = SyntheticControl()
        assert isinstance(sc, SyntheticControl)

    def test_setup_model(self):
        sc = SyntheticControl(fit_intercept=True, force_positive=False)
        sc.setup_model()
        assert isinstance(sc.model, Ridge)
        assert sc.model.fit_intercept is True
        assert sc.model.positive is False
