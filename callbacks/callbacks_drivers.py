import time
import numpy as np
from dash import Input, Output, State, html, dcc, ALL, ctx, no_update

from util.plotters import arrange_corr_matx
from util.analysis import run_rfe
from _config import *


def register_callbacks_drivers(app, df, drivers, production=True):

    @app.callback(
        Output("analysis-btn", "disabled", allow_duplicate=True),
        Input("analysis-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def disable_btn(n):
        if n:
            return True
        return no_update

    @app.callback(
        Output("drivers-results-container", "children"),
        Output("analysis-btn", "disabled"),
        Input("analysis-btn", "n_clicks"),
        Input("tab-content", "children"),
        State("rfe-metric-dropdown", "value"),
        State({"type": "driver-checkbox", "index": ALL}, "value"),
        State({"type": "driver-checkbox", "index": ALL}, "id"),
        prevent_initial_call=False,
    )
    def _run_analysis_and_write_token(n_clicks, tab_children, metric, values, ids):

        triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else None

        # Get predictors
        predictors = [id_["index"] for val, id_ in zip(values, ids) if val]

        # On load: correlation
        if not predictors or not triggered.startswith("analysis-btn"):
            predictors = [id_["index"] for val, id_ in zip(values, ids) if val]
            if not predictors:
                return (
                    html.P("No explanatory variables selected for correlation matrix."),
                    False,
                )
            corr_vars = [
                c
                for c in predictors
                if c in df.columns and np.issubdtype(df[c].dtype, np.number)
            ]
            if not corr_vars:
                return (
                    html.P("No numeric variables available for correlation matrix."),
                    False,
                )
            fig_corr = arrange_corr_matx(df[corr_vars])
            return html.Div([html.H4("Correlation analysis"), fig_corr]), False

        # On 'Run analysis' click: RFE
        if not metric or not predictors:
            return (
                html.P(
                    "Please select a metric and predictors before running analysis."
                ),
                False,
            )
        sub = df[[metric] + predictors].dropna()
        n_sample = len(sub)
        if n_sample < MIN_EVENT:
            return (
                html.Div(
                    f"Insufficient events ({n_sample}, need minimum {MIN_EVENT})."
                ),
                False,
            )

        try:
            summary, plot_feature, plot_pdp = run_rfe(drivers, sub, metric, predictors)
        except Exception as e:
            return html.Div([html.P("Error running analysis"), html.Pre(str(e))]), False

        results = html.Div(
            [
                html.H4("Recursive feature elimination"),
                html.Div(summary),
                html.Hr(),
                html.H4("Feature importance"),
                plot_feature,
                html.Hr(),
                html.H4("Partial dependence plots"),
                html.Div(plot_pdp),
            ]
        )

        return results, False
