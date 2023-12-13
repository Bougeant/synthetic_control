# -*- coding: utf-8 -*-

""" Functions to provide analysis for the synthetic control method. """

import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS


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
    treatment_color = DEFAULT_PLOTLY_COLORS[0]
    control_color = DEFAULT_PLOTLY_COLORS[1]
    data = [
        go.Scatter(x=y.index, y=y, name=treatment_name, line_color=treatment_color),
        go.Scatter(
            x=y.index, y=y_pred, name="Synthetic Control", line_color=control_color
        ),
    ]
    data = _add_confidence_interval(data, y_pred_ci, control_color)
    return data


def _add_confidence_interval(data, y_pred_ci, color):
    """ """
    min_ci = y_pred_ci.columns.min()
    max_ci = y_pred_ci.columns.max()
    ci_range = max_ci - min_ci
    color = _get_opacity_color(color)
    for i, col in enumerate(y_pred_ci.columns):
        data.append(
            go.Scatter(
                x=y_pred_ci.index,
                y=y_pred_ci[col],
                name=f"{ci_range}% confidence interval",
                fill="tonexty",
                line_color=color,
                fillcolor=color,
                showlegend=i == 0,
            )
        )
    return data


def _get_opacity_color(color):
    return color.replace("rgb", "rgba").replace(")", ", 0.1)")


def _get_plot_layout(y_axis):
    """ """
    layout = go.Layout(
        xaxis={"title": "Date"},
        yaxis={"title": y_axis},
    )
    return layout
