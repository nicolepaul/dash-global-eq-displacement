import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
import dash_bootstrap_components as dbc

import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import medianabs, medianbias
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, root_mean_squared_error, r2_score
from sklearn.feature_selection import RFECV
from sklearn.inspection import partial_dependence

from _config import *
from util.plotters import plot_model_eval


def transform_variables(df, drivers):
    for i, row in drivers.iterrows():
        if row["transform"] == "log":
            df[row.variable] = np.log1p(df[row.variable])
            drivers.loc[i, "name"] = "log(" + drivers.loc[i, "name"] + ")"
    return df, drivers


def run_regression(df, x_col, y_col, method="ols", add_trace=True):
    # TODO: Include some evaluation metric or visual to indicate predictive performance beyond R2

    N = 200

    df["log_x"] = np.log1p(df[x_col])
    df["log_y"] = np.log1p(df[y_col])

    eqn = None
    lm_fit = None
    color, style = "black", "dash"
    if method == "ols":
        lm = smf.ols(formula="log_y ~ log_x - 1", data=df)
        lm_fit = lm.fit()
        eqn = html.Div(
            [
                dcc.Markdown("$\\log y=\\beta \\cdot \\log x$", mathjax=True),
                dcc.Markdown("Or, $y = x^\\beta$", mathjax=True),
            ]
        )
    elif method == "ols_int":
        lm = smf.ols(formula="log_y ~ log_x", data=df)
        lm_fit = lm.fit()
        eqn = html.Div(
            [
                dcc.Markdown("$\\log y=\\beta \\cdot \\log x + \\alpha$", mathjax=True),
                dcc.Markdown("Or, $y = e^{\\alpha}x^{\\beta}$", mathjax=True),
            ]
        )
    elif method == "rlm":
        lm = smf.rlm(formula="log_y ~ log_x - 1", data=df)
        lm_fit = lm.fit()
        eqn = html.Div(
            [
                dcc.Markdown("$\\log y=\\beta \\cdot \\log x$", mathjax=True),
                dcc.Markdown("Or, $y = x^\\beta$", mathjax=True),
            ]
        )
        color, style = "darkred", "dot"
    elif method == "rlm_int":
        lm = smf.rlm(formula="log_y ~ log_x", data=df)
        lm_fit = lm.fit()
        eqn = html.Div(
            [
                dcc.Markdown("$\\log y=\\beta \\cdot \\log x + \\alpha$", mathjax=True),
                dcc.Markdown("Or, $y = e^{\\alpha}x^{\\beta}$", mathjax=True),
            ]
        )
        color, style = "darkred", "dot"
    else:
        raise NotImplementedError(f"Method '{method}' not supported yet.")

    # Get key values
    alpha = lm_fit.params["Intercept"] if hasattr(lm_fit.params, "Intercept") else None
    beta = lm_fit.params["log_x"] if hasattr(lm_fit.params, "log_x") else None
    r2 = lm_fit.rsquared if hasattr(lm_fit, "rsquared") else None
    r2_adj = lm_fit.rsquared_adj if hasattr(lm_fit, "rsquared_adj") else None
    ypred = lm_fit.predict(df['log_x'])
    mdae = medianabs(df['log_y'], ypred)
    bias = medianbias(df['log_y'], ypred)

    # Write narrative
    narrative = []
    if beta is not None:

        regression_results = dbc.ListGroup(
            [
                dbc.ListGroupItem(f"n = {df[y_col].count():,.0f}"),
                dbc.ListGroupItem(f"α = {alpha:.3f}" if alpha is not None else ""),
                dbc.ListGroupItem(f"β = {beta:.3f}" if beta is not None else ""),
                dbc.ListGroupItem(
                    f"R² = {r2:.3f}" if r2 is not None else "R² not available"
                ),
                dbc.ListGroupItem(
                    f"R² (adj.) = {r2_adj:.3f}" if r2_adj is not None else "R² (adj.) not available"
                ),
                dbc.ListGroupItem(f"MdAPE = {mdae:.1%}"),
                dbc.ListGroupItem(f"Median bias = {bias:.1%}"),
            ],
            flush=True,
        )
        narrative = [
            html.P(),
            eqn,
            html.B("Regression Results"),
            regression_results,
        ]

    # Define trace
    trace = None
    if add_trace:
        log_x = np.linspace(np.log1p(FILL_ZERO), df["log_x"].max(), N)
        y_hat = lm_fit.predict(exog=dict(log_x=log_x))

        trace = go.Scatter(
            x=np.expm1(log_x),
            y=np.expm1(y_hat),
            mode="lines",
            name=f"{method.upper()} (β={beta:.3f})",
            line=dict(color=color, dash=style),
            hoverinfo="skip",
            zorder=5,
        )

    return trace, narrative


def mape_from_log(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mask = y_true != 0
    return np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def run_rfe(drivers, sub, metric, predictors, production=True):

    X, y = sub[predictors], np.log1p(sub[metric])

    # Tune hyperparameters
    params = None
    mape_scorer = make_scorer(mape_from_log, greater_is_better=False)
    if production:  # hard-coded to save computation time
        if metric == "sheltered_peak":
            params = {
                "n_estimators": 50,
                "max_depth": 3,
                "learning_rate": 0.2,
                "reg_alpha": 0.1, 
                "reg_lambda": 1, 
                "gamma": 0.3,
                "min_child_weight": 3, 
            } 
        elif metric == "evacuated":
            params = {
                "n_estimators": 50,
                "max_depth": 2,
                "learning_rate": 0.3,
                "reg_alpha": 0.1,
                "reg_lambda": 1,
                "gamma": 0.3,
                "min_child_weight": 5,
            } 
        elif metric == "protracted":
            params = {
                "n_estimators": 50,
                "max_depth": 3,
                "learning_rate": 0.3,
                "reg_alpha": 0.1,
                "reg_lambda": 1,
                "gamma": 0.3,
                "min_child_weight": 5,
            } 
        elif metric == "assisted":
            params = {
                "n_estimators": 50,
                "max_depth": 2,
                "learning_rate": 0.3,
                "reg_alpha": 0.1,
                "reg_lambda": 1,
                "gamma": 0.3,
                "min_child_weight": 5,
            } 
        else:
            return (
                f"Production version hyperparameters not configured for {metric}",
                EMPTY_FIG,
                [],
            )

    else:
        grid = GridSearchCV(
            XGBRegressor(random_state=99),
            param_grid={
                "n_estimators": [50],
                "max_depth": [2, 3, 4],
                "learning_rate": [0.1, 0.2, 0.3],
                "reg_alpha": [0.1, 1],
                "reg_lambda": [1, 10],
                "gamma": [0.1, 0.3, 0.5],
                "min_child_weight": [3, 5],
            },
            cv=CV,
            scoring=mape_scorer,
            n_jobs=1,
        )
        grid.fit(X, y) 
        params = grid.best_params_
    parm = dbc.Row(
        [
            html.B("Model hyperparameters:"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem(
                        f"Number of estimators = {params['n_estimators']}"
                        if params["n_estimators"] is not None
                        else ""
                    ),
                    dbc.ListGroupItem(
                        f"Max depth = {params['max_depth']}"
                        if params["max_depth"] is not None
                        else ""
                    ),
                    dbc.ListGroupItem(
                        f"Learning rate = {params['learning_rate']}"
                        if params["learning_rate"] is not None
                        else ""
                    ),
                    dbc.ListGroupItem(
                        f"Regularization α = {params['reg_alpha']}"
                        if params["reg_alpha"] is not None
                        else ""
                    ),
                    dbc.ListGroupItem(
                        f"Regularization λ = {params['reg_lambda']}"
                        if params["reg_lambda"] is not None
                        else ""
                    ),
                    dbc.ListGroupItem(
                        f"Gamma = {params['gamma']}"
                        if params["gamma"] is not None
                        else ""
                    ),
                    dbc.ListGroupItem(
                        f"Min child weight = {params['min_child_weight']}"
                        if params["min_child_weight"] is not None
                        else ""
                    ),
                ],
                flush=True,
            ),
        ]
    )

    # Use CV with RFE to select features
    model = XGBRegressor(**params)
    cv = KFold(n_splits=CV, shuffle=True, random_state=22)
    if not production:
        cv = RepeatedKFold(n_splits=CV, n_repeats=S, random_state=22)
    rfecv = RFECV(model, cv=cv, scoring=mape_scorer, step=1, n_jobs=1)
    rfecv.fit(X, y)
    selected = list(X.columns[rfecv.support_])

    # Arrange text summary
    summ = dbc.Row(
        [
            html.P(),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem(f"Number of events: {len(sub)}"),
                    dbc.ListGroupItem(f"Number of cross-validation folds: {CV}"),
                ]
            ),
            html.P(),
            html.B(f"Selected {len(selected)} feature(s):"),
            dbc.ListGroup(
                [dbc.ListGroupItem(sel) for sel in selected],
                flush=True,
            ),
        ]
    )

    # Refit model if not in production mode
    final_model = XGBRegressor(**params)
    final_model.fit(X[selected], y)
    importances = final_model.feature_importances_

    # Evaluate model
    y_pred = final_model.predict(X[selected])
    rmse_log = root_mean_squared_error(y, y_pred)
    r2_log = r2_score(y, y_pred)
    mdape = mape_from_log(y, y_pred)
    bias = np.median(y_pred - y)
    eval = dbc.Row(
        [
            html.B("Model performance:"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem(
                        f"MdAPE = {mdape:.1%}" if mdape is not None else ""
                    ),
                    dbc.ListGroupItem(
                        f"RMSE (log) = {rmse_log:.3f}" if rmse_log is not None else ""
                    ),
                    dbc.ListGroupItem(
                        f"R² (log) = {r2_log:.3f}" if r2_log is not None else ""
                    ),
                    dbc.ListGroupItem(
                        f"Median bias (log) = {bias:.1%}" if bias is not None else ""
                    ),
                ],
                flush=True,
            ),
        ]
    )
    fig_eval_log = plot_model_eval(
        y, y_pred
    )  # .update_layout(title='Model evaluation (log)')
    fig_eval = plot_model_eval(
        np.expm1(y), np.expm1(y_pred)
    )  # .update_layout(title='Model evaluation')

    # Feature importance plot
    fig_imp = go.Figure(
        [
            go.Bar(
                x=importances,
                y=selected,
                orientation="h",
                hovertemplate="<b>%{y}: </b>%{x:.1%}<extra></extra>",
            )
        ]
    )
    fig_imp.update_layout(yaxis=dict(autorange="reversed"))
    fig_feature = dcc.Graph(figure=fig_imp, style={"height": "400px"})

    # Partial dependence plots; TODO: improve tooltip, consider markers
    pdp_children = []
    if not selected:
        return (
            [html.P("No features were selected. Try different predictors.")],
            fig_imp,
            [],
        )
    else:
        for feat in selected:
            pred = partial_dependence(final_model, X[selected], feat, kind='both')
            ice_curves, grid = np.squeeze(pred['individual']), pred['grid_values']

            fig = go.Figure()

            for sample_curve in ice_curves:
                fig.add_trace(go.Scatter(x=grid[0], y=sample_curve, mode='lines', line=dict(width=1, color='silver'), hoverinfo="skip"))

            avg = ice_curves.mean(axis=0)
            fig.add_trace(go.Scatter(x=grid[0], y=avg, mode='lines', line=dict(width=3, color="#212121"), name='Average'))

            fig.update_layout(
                title=drivers.loc[drivers.variable == feat, "name"].values[0],
                showlegend=False,
                margin=dict(l=40, r=20, t=30, b=30),
                height=300,
            )

            pdp_children.append(dcc.Graph(figure=fig, style={"height": "300px"}))

    summary = [
        dbc.Row(
            [
                dbc.Col([summ, html.P(), parm, html.P(), eval]),
                dbc.Col(dcc.Graph(figure=fig_eval)),
            ]
        ),
    ]

    return summary, fig_feature, pdp_children
