# callbacks/callbacks_drivers.py
from dash import Input, Output, State, html, dcc
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from sklearn.inspection import partial_dependence
from _config import *


def register_callbacks_drivers(app, df):

    @app.callback(
        Output("rfe-summary", "children"),
        Output("rfe-feature-importance", "figure"),
        Output("rfe-pdp-container", "children"),
        Input("rfe-run-btn", "n_clicks"),
        State("rfe-metric-dropdown", "value"),
        State("rfe-predictors-dropdown", "value"),
        prevent_initial_call=True,
    )
    def run_rfe(n_clicks, metric, predictors):
        if not metric or not predictors:
            return "Please select a metric and predictors.", EMPTY_FIG, []

        sub = df[[metric] + predictors].dropna()
        if len(sub) < MIN_EVENT:
            return f"Insufficient events ({len(sub)}, need minimum {MIN_EVENT}).", EMPTY_FIG, []

        X, y = sub[predictors], sub[metric]
        params = {
                    'learning_rate': 0.02,
                    'max_depth': 3,
                    'n_estimators': 100,
                    'random_state': 99,
                    'n_jobs': 1
                }
        model = XGBRegressor(**params)
        cv = KFold(n_splits=CV, shuffle=True, random_state=0)
        rfecv = RFECV(model, cv=cv, scoring="r2", n_jobs=-1)
        rfecv.fit(X, y)

        selected = list(X.columns[rfecv.support_])
        final_model = rfecv.estimator_
        importances = final_model.feature_importances_

        # Feature importance bar chart
        fig_imp = go.Figure([go.Bar(x=importances, y=selected, orientation="h")])
        fig_imp.update_layout(
            title="Feature Importances",
            yaxis=dict(autorange="reversed"),
        )

        # Partial dependence plots
        pdp_children = []
        if not selected:
            return [
                html.P("No features were selected. Try different predictors.")
            ], fig_imp, []
        else:
            for feat in selected:
                pd_vals = partial_dependence(final_model, X[selected], [feat])
                grid = pd_vals["values"][0]
                vals = pd_vals["average"][0]
                fig = go.Figure(go.Scatter(x=grid, y=vals, mode="lines"))
                fig.update_layout(title=f"PDP: {feat}")
                pdp_children.append(dcc.Graph(figure=fig, style={"height": "300px"}))

        summary = [
            html.P(f"Optimal features: {len(selected)}"),
            html.P(f"Selected: {', '.join(selected)}"),
        ]

        return summary, fig_imp, pdp_children
