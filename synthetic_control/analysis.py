# -*- coding: utf-8 -*-

""" Functions to provide analysis for the synthetic control method. """

import plotly.graph_objects as go


def display_synthetic_control(y, y_pred, y_pred_ci, treatment_start, **kwargs):
    """ """
    data = _get_plot_data(y, y_pred, y_pred_ci, treatment_start, **kwargs)
    layout = _get_plot_layout()
    return go.Figure(data=data, layout=layout)


def _get_plot_data(
    y,
    y_pred,
    y_pred_ci,
    treatment_start,
    treatment_end=None,
    treatment_name="Treatment",
):
    """ """
    data = [
        go.Scatter(x=y.index, y=y),
        go.Scatter(x=y.index, y=y_pred),
    ]
    return data
