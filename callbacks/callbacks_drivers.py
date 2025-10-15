# callbacks/callbacks_drivers.py
from dash import Input, Output, State, html, dcc, ALL, MATCH, no_update
import numpy as np
import plotly.graph_objs as go
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV
from sklearn.metrics import make_scorer, root_mean_squared_error, r2_score
from sklearn.feature_selection import RFECV
from sklearn.inspection import partial_dependence
from _config import *
from util.analysis import mape_from_log


def register_callbacks_drivers(app, df, drivers):


    @app.callback(
        Output("rfe-summary", "children"),
        Output("rfe-feature-importance", "children"),
        Output("rfe-pdp-container", "children"),
        Input("analysis-btn", "n_clicks"),
        State("rfe-metric-dropdown", "value"),
        State({"type": "driver-checkbox", "index": ALL}, "value"),
        State({"type": "driver-checkbox", "index": ALL}, "id"),
        prevent_initial_call=True,
    )
    def run_rfe(n_clicks, metric, ids, values):

        # Parse inputs
        predictors = [val["index"] for b, val in zip(ids, values) if b]
        if not metric or not predictors:
            return "Please select a metric and predictors.", EMPTY_FIG, []
        sub = df[[metric] + predictors].dropna()
        n_sample = len(sub)
        if n_sample < MIN_EVENT:
            return (
                f"Insufficient events ({len(sub)}, need minimum {MIN_EVENT}).",
                EMPTY_FIG,
                [],
            )
        X, y = sub[predictors], np.log1p(sub[metric])

        # Tune hyperparameters
        mape_scorer = make_scorer(mape_from_log, greater_is_better=False)
        grid = GridSearchCV(
        XGBRegressor(random_state=99),
        param_grid={"max_depth": [2, 3, 4], "learning_rate": [0.01, 0.02, 0.1, 0.2], "min_child_weight": [2, 3]},
            cv=CV, scoring=mape_scorer, n_jobs=1
        )
        grid.fit(X, y)
        params = grid.best_params_
        parm = [html.B('Model hyperparameters:'),
                dbc.ListGroup(
            [
                dbc.ListGroupItem(f"Max depth = {params['max_depth']}" if params['max_depth'] is not None else ""),
                dbc.ListGroupItem(f"Learning rate = {params['learning_rate']}" if params['learning_rate'] is not None else ""),
                dbc.ListGroupItem(
                    f"Min child weight = {params['min_child_weight']}" if params['min_child_weight'] is not None else ""
                ),
            ],
            flush=True,
        )]

        # Use CV with RFE to select features
        # params = {
        #     "learning_rate": 0.02,
        #     "max_depth": 3,
        #     "n_estimators": 100,
        #     "random_state": 99,
        #     "n_jobs": 1,
        #     "min_child_weight": 2, 
        # }
        model = XGBRegressor(**params)
        # cv = KFold(n_splits=CV, shuffle=True, random_state=22)
        cv = RepeatedKFold(n_splits=CV, n_repeats=5, random_state=22)
        rfecv = RFECV(model, cv=cv, scoring=mape_scorer, step=1, n_jobs=1)
        rfecv.fit(X, y)
        selected = list(X.columns[rfecv.support_])

        # Refit model
        final_model = XGBRegressor(**params)
        final_model.fit(X[selected], y)
        importances = final_model.feature_importances_

        # Evaluate model
        final_model.fit(X[selected], y)
        y_pred = final_model.predict(X[selected])
        rmse_log = root_mean_squared_error(y, y_pred)
        r2_log = r2_score(y, y_pred)
        mdape = mape_from_log(y, y_pred)
        eval = [html.B('Model performance:'),
                dbc.ListGroup(
            [
                dbc.ListGroupItem(f"MdAPE = {mdape:.3f}" if mdape is not None else ""),
                dbc.ListGroupItem(f"RMSE (log) = {rmse_log:.3f}" if rmse_log is not None else ""),
                dbc.ListGroupItem(
                    f"R² (log) = {r2_log:.3f}" if r2_log is not None else ""
                ),
            ],
            flush=True,
        )]

        # Feature importance plot; TODO: improve tooltip, show percentages
        fig_imp = go.Figure([go.Bar(x=importances, y=selected, orientation="h")])
        fig_imp.update_layout(
            yaxis=dict(autorange="reversed"),
        )
        fig_imp_div = dcc.Graph(figure=fig_imp, style={"height": "300px"})

        # Partial dependence plots; TODO: improve tooltip, consider markers
        pdp_children = []
        if not selected:
            return [
                html.P("No features were selected. Try different predictors.")
            ], fig_imp, []
        else:
            for feat in selected:
                pred = partial_dependence(final_model, X[selected], feat)
                avg, grid = pred['average'], pred['grid_values']

                fig = go.Figure(
                    go.Scatter(x=grid[0], y=avg[0], mode="lines", name=f"PDP: {feat}")
                )
                fig.update_layout(
                    title=drivers.loc[drivers.variable==feat, 'name'].values[0],
                    margin=dict(l=40, r=20, t=30, b=30),
                    height=300,
                )

                pdp_children.append(
                    dcc.Graph(figure=fig, style={"height": "300px"})
                )

        summary = [
            html.P(f"Number of events: {n_sample}"),
            html.P(f"Selected {len(selected)} features: {', '.join(selected)}"),
            dbc.Row([dbc.Col(parm), dbc.Col(eval)])

        ]

        return summary, fig_imp_div, pdp_children
