# -*- coding: utf-8 -*-

""" Tests for synthetic_control.analysis.compare. """

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from synthetic_control.analysis import compare


class TestCompare:
    def tests_compare_to_synthetic_control(self):
        """Test the compare_to_synthetic_control function."""
        y = pd.Series([100, 90, 80, 120, 100, 90, 100])
        y_pred_ci = pd.DataFrame(
            {
                5: [96.8, 81.2, 71.4, 51.4, 86.5, 70.7, 82.3],
                95: [118.1, 101.6, 100.1, 169.9, 121.1, 140.7, 98.2],
            },
            index=y.index,
        )
        fig = compare.compare_to_synthetic_control(
            y, y_pred_ci, datetime(2008, 1, 1), datetime(2010, 1, 1)
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4
        assert fig.data[0].name == "Treatment"
        assert fig.data[1].name == "Synthetic Control"
        assert fig.data[2].name == "90% confidence interval"
        assert fig.data[3].name == "90% confidence interval"

    def test_get_plot_data(self):
        """Test the get_plot_data function."""
        y = pd.Series([100, 90, 80, 120, 100, 90, 100])
        y_pred_ci = pd.DataFrame(
            {
                5: [96.8, 81.2, 71.4, 51.4, 86.5, 70.7, 82.3],
                95: [118.1, 101.6, 100.1, 169.9, 121.1, 140.7, 98.2],
            },
            index=y.index,
        )
        data = compare.get_plot_data(y, y_pred_ci, "Treatment")
        assert len(data) == 4
        assert all([isinstance(d, go.Scatter) for d in data])

    def test_get_baseline_prediction_with_median(self):
        """Test the get_baseline_prediction function with median."""
        y_pred_ci = pd.DataFrame({10: [10, 11, 12], 50: [20, 21, 22], 90: [26, 28, 28]})
        y_pred = compare.get_baseline_prediction(y_pred_ci)
        assert isinstance(y_pred, pd.Series)
        assert y_pred.equals(pd.Series([20, 21, 22]))

    def test_get_baseline_prediction_without_median(self):
        """Test the get_baseline_prediction function with median."""
        y_pred_ci = pd.DataFrame({10: [10, 11, 12], 45: [21, 21, 23], 90: [26, 28, 28]})
        y_pred = compare.get_baseline_prediction(y_pred_ci)
        assert isinstance(y_pred, pd.Series)
        assert y_pred.equals(pd.Series([19.0, 20.0, 21.0]))
