# -*- coding: utf-8 -*-

""" Tests for synthetic_control.analysis.compare. """


from datetime import datetime

import pandas as pd
import plotly.graph_objects as go

from synthetic_control.analysis import compare


class TestCompare:
    def tests_compare_to_synthetic_control(self):
        """Test the compare_to_synthetic_control function."""
        y = pd.Series([100, 90, 80])
        y_pred_ci = pd.DataFrame({5: [96, 81, 73], 95: [118, 101, 100]})
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
        y = pd.Series([100, 90, 80])
        y_pred_ci = pd.DataFrame({5: [96, 81, 73], 95: [118, 101, 100]})
        data = compare.get_plot_data(y, y_pred_ci, "Treatment", show_impact=False)
        assert len(data) == 4
        assert all([isinstance(d, go.Scatter) for d in data])
        assert list(data[0].y) == [100, 90, 80]
        assert list(data[-1].y) == [118, 101, 100]

    def test_get_plot_data_impact(self):
        """Test the get_plot_data function."""
        y = pd.Series([100, 90, 80])
        y_pred_ci = pd.DataFrame({5: [96, 81, 73], 95: [118, 101, 100]})
        data = compare.get_plot_data(y, y_pred_ci, "Treatment", show_impact=True)
        assert len(data) == 4
        assert all([isinstance(d, go.Scatter) for d in data])
        assert list(data[0].y) == [0, 0, 0]
        assert list(data[-1].y) == [18, 11, 20]

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

    def test_add_confidence_interval(self):
        """Test the add_confidence_interval function."""
        y_pred_ci = pd.DataFrame(
            {5: [10, 11, 12], 25: [13, 15, 15], 75: [18, 19, 20], 95: [20, 21, 22]}
        )
        data = compare.add_confidence_interval(y_pred_ci, "red")
        assert len(data) == 4
        assert [d.name for d in data] == ["90% confidence interval"] * 4
        assert all([isinstance(d, go.Scatter) for d in data])

    def test_get_opacity_color(self):
        """Test the get_opacity_color function."""
        color = "rgb(140, 20, 14)"
        assert compare.get_opacity_color(color, opacity=0.5) == "rgba(140, 20, 14, 0.5)"

    def test_add_treatment_period(self):
        """Test the add_treatment_period function."""
        fig = go.Figure()
        fig = compare.add_treatment_period(
            fig, datetime(2008, 1, 1), datetime(2010, 1, 1)
        )
        assert len(fig.layout.shapes) == 2
        treatment_start = datetime.fromtimestamp(fig.layout.shapes[0].x0 / 1000)
        treatment_end = datetime.fromtimestamp(fig.layout.shapes[1].x0 / 1000)
        assert treatment_start == datetime(2008, 1, 1)
        assert treatment_end == datetime(2010, 1, 1)

    def test_get_plot_layout(self):
        """Test the get_plot_layout function."""
        layout = compare.get_plot_layout(y_axis="Test value", show_impact=False)
        assert isinstance(layout, go.Layout)
        assert layout.xaxis.title.text == "Date"
        assert layout.yaxis.title.text == "Test value"

    def test_get_plot_layout_impact(self):
        """Test the get_plot_layout function."""
        layout = compare.get_plot_layout(y_axis="Test value", show_impact=True)
        assert isinstance(layout, go.Layout)
        assert layout.xaxis.title.text == "Date"
        assert layout.yaxis.title.text == "Difference in Test value"
