# -*- coding: utf-8 -*-

""" Functions to provide analysis for the synthetic control method. """

import plotly.graph_objects as go


def display_synthetic_control(
    y, y_pred, y_pred_ci, treatment_start, y_axis="Value", **kwargs
):
    """ """
    data = _get_plot_data(y, y_pred, y_pred_ci, treatment_start, **kwargs)
    layout = _get_plot_layout(y_axis)
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
        go.Scatter(x=y.index, y=y, name=treatment_name),
        go.Scatter(x=y.index, y=y_pred, name="Synthetic Control"),
    ]
    return data


def _get_plot_layout(y_axis):
    """ """
    layout = go.Layout(
        xaxis={"title": "Date"},
        yaxis={"title": y_axis},
    )
    return layout
