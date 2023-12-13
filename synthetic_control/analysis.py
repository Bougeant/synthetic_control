# -*- coding: utf-8 -*-

""" Functions to provide analysis for the synthetic control method. """

from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS


def display_synthetic_control(
    y,
    y_pred_ci,
    treatment_start,
    treatment_end=None,
    treatment_name="Treatment",
    y_axis="Value",
):
    """ """
    data = get_plot_data(y, y_pred_ci, treatment_name)
    layout = get_plot_layout(y_axis)
    fig = go.Figure(data=data, layout=layout)
    fig = add_treatment_period(fig, treatment_start, treatment_end)
    return fig


def get_plot_data(y, y_pred_ci, treatment_name):
    """ """
    treatment_color = DEFAULT_PLOTLY_COLORS[0]
    control_color = DEFAULT_PLOTLY_COLORS[1]
    y_pred = get_base_prediction(y_pred_ci)
    data = [
        go.Scatter(
            x=y.index,
            y=y,
            name=treatment_name,
            line_color=treatment_color,
            line_width=5,
        ),
        go.Scatter(
            x=y.index,
            y=y_pred,
            name="Synthetic Control",
            line_color=control_color,
            line_width=5,
        ),
    ]
    data = add_confidence_interval(data, y_pred_ci, control_color)
    return data


def get_base_prediction(y_pred_ci):
    """ """
    if 50 in y_pred_ci:
        y_pred = y_pred_ci[50]
    else:
        y_pred = y_pred_ci.mean(axis=1)
    return y_pred


def add_confidence_interval(data, y_pred_ci, color):
    """ """
    min_ci = y_pred_ci.columns.min()
    max_ci = y_pred_ci.columns.max()
    ci_range = max_ci - min_ci
    color = get_opacity_color(color)
    columns = y_pred_ci.columns.sort_values(
        key=lambda x: np.abs(50 - x), ascending=False
    )
    for i, col in enumerate(columns):
        data.append(
            go.Scatter(
                x=y_pred_ci.index,
                y=y_pred_ci[col],
                name=f"{ci_range}% confidence interval",
                fill="tonexty",
                line_color=color,
                fillcolor=color,
                showlegend=i == 0,
                legendgroup="CI",
            )
        )
    return data


def get_opacity_color(color):
    return color.replace("rgb", "rgba").replace(")", ", 0.1)")


def add_treatment_period(fig, treatment_start, treatment_end):
    """ """
    fig.add_vline(
        x=treatment_start.timestamp() * 1000,
        line_color="black",
        line_dash="dash",
        annotation={"text": "Treatment Start", "xanchor": "center", "y": 1.1},
    )
    if treatment_end:
        fig.add_vline(
            x=treatment_end.timestamp() * 1000,
            line_color="black",
            line_dash="dash",
            annotation={"text": "Treatment End", "xanchor": "center", "y": 1.1},
        )
    return fig


def get_plot_layout(y_axis):
    """ """
    layout = go.Layout(xaxis={"title": "Date"}, yaxis={"title": y_axis})
    return layout
