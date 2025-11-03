import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html
import dash_bootstrap_components as dbc

import statsmodels.formula.api as smf

from xgboost import XGBRegressor
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    GridSearchCV,
)
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.feature_selection import RFECV

import shap


from _config import *
from util.metrics import (
    rsquared,
    percent_error,
    percent_absolute_error,
    custom_score,
)
from util.plotters import (
    plot_model_eval,
    plot_feature_importance,
    plot_pdp,
    plot_interactions,
)


def transform_variables(df, drivers):
    for i, row in drivers.iterrows():
        if row["transform"] == "log":
            df[row.variable] = np.log(df[row.variable].replace(0, FILL_ZERO))
            drivers.loc[i, "name"] = "log(" + drivers.loc[i, "name"] + ")"
    return df, drivers


def run_regression(df, x_col, y_col, method="ols", add_trace=True):
    N = 200

    df["log_x"] = np.log(df[x_col])
    df["log_y"] = np.log(df[y_col])

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
    ypred = lm_fit.predict(df["log_x"])

    # Write narrative
    narrative = []
    if beta is not None:

        regression_results = dbc.ListGroup(
            [
                dbc.ListGroupItem(f"n = {df[y_col].count():,.0f}"),
                dbc.ListGroupItem(f"α = {alpha:.3f}" if alpha is not None else ""),
                dbc.ListGroupItem(f"β = {beta:.3f}" if beta is not None else ""),
                dbc.ListGroupItem(f"R² = {rsquared(df['log_y'], ypred):.1%}"),
                dbc.ListGroupItem(
                    f"Median abs. percent error = {np.median(percent_absolute_error(df['log_y'], ypred)):.1%}"
                ),
                dbc.ListGroupItem(
                    f"Mean abs. percent error = {np.mean(percent_absolute_error(df['log_y'], ypred)):.1%}"
                ),
                dbc.ListGroupItem(
                    f"Consensus score = {custom_score(df['log_y'], ypred):.1%}"
                ),
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
        log_x = np.linspace(np.log(FILL_ZERO), df["log_x"].max(), N)
        y_hat = lm_fit.predict(exog=dict(log_x=log_x))

        trace = go.Scatter(
            x=np.exp(log_x),
            y=np.exp(y_hat),
            mode="lines",
            name=f"{method.upper()} (β={beta:.3f})",
            line=dict(color=color, dash=style),
            hoverinfo="skip",
            zorder=5,
        )

    return trace, narrative


def summarize_model(params):
    if params is None:
        return html.Div()
    items = []
    items.append(
        dbc.ListGroupItem(f"Number of estimators = {params.get('n_estimators','')}")
    )
    items.append(dbc.ListGroupItem(f"Max depth = {params.get('max_depth','')}"))
    items.append(dbc.ListGroupItem(f"Learning rate = {params.get('learning_rate','')}"))
    items.append(dbc.ListGroupItem(f"Gamma = {params.get('gamma','')}"))
    items.append(
        dbc.ListGroupItem(f"Min child weight = {params.get('min_child_weight','')}")
    )
    return dbc.Row([html.B("Model hyperparameters:"), dbc.ListGroup(items, flush=True)])


def summarize_evaluation(y, y_pred):
    try:
        eval_items = [
            dbc.ListGroupItem(f"Median abs. percent error = {np.median(percent_error(y, y_pred)):.1%}"),
            dbc.ListGroupItem(f"Mean abs. percent error = {np.mean(percent_error(y, y_pred)):.1%}"),
            dbc.ListGroupItem(f"Root mean square error = {root_mean_squared_error(y, y_pred):.3f}"),
            dbc.ListGroupItem(f"R² = {rsquared(y, y_pred):.1%}"),
            dbc.ListGroupItem(f"Consensus score = {custom_score(y, y_pred):.1%}"),
        ]
    except Exception as e:
        eval_items = [dbc.ListGroupItem(f"Error computing metrics: {e}")]

    return dbc.Row([dbc.ListGroup(eval_items, flush=True)])


def compute_interactions(model, X, features):
    explainer = shap.TreeExplainer(model)
    interaction_values = explainer.shap_interaction_values(X[features])
    mean_abs_interactions = np.abs(interaction_values).mean(axis=0)

    interaction_strength = pd.DataFrame(
        mean_abs_interactions, index=features, columns=features
    )
    np.fill_diagonal(interaction_strength.values, np.nan)
    return interaction_strength


def run_rfe(drivers, sub, metric, predictors, production=True):

    # Arrange response and explanatory variables
    X, y = sub[predictors], np.log(sub[metric].replace(0, FILL_ZERO))

    # Tune hyperparameters
    params = None
    mape_scorer = make_scorer(custom_score, greater_is_better=False)
    if production:  # hard-coded to save computation time
        if metric in PARAM_PROD:
            params = PARAM_PROD[metric]

        else:
            return (
                f"Production version hyperparameters not configured for {metric}",
                EMPTY_FIG,
                [],
            )

    else:
        grid = GridSearchCV(
            XGBRegressor(random_state=MODEL_SEED),
            param_grid=PARAM_GRID,
            cv=CV,
            scoring=mape_scorer,
            n_jobs=1,
        )
        grid.fit(X, y)
        params = grid.best_params_
    parm = summarize_model(params)

    # Use CV with RFE to select features; use repeats if not in production mode
    model = XGBRegressor(random_state=MODEL_SEED, **params)
    cv = KFold(n_splits=CV, shuffle=True, random_state=CV_SEED)
    if not production:
        cv = RepeatedKFold(n_splits=CV, n_repeats=S, random_state=CV_SEED)
    rfecv = RFECV(model, cv=cv, scoring=mape_scorer, step=1, n_jobs=1)
    rfecv.fit(X, y)
    selected = list(X.columns[rfecv.support_])

    # Arrange text summary
    summ = dbc.Row(
        [
            html.B(f"Selected {len(selected)} feature(s):"),
            dbc.ListGroup(
                [dbc.ListGroupItem(sel) for sel in selected],
                flush=True,
            ),
        ]
    )

    # Get out-of-fold predictions
    cv = KFold(n_splits=CV, shuffle=True, random_state=CV_SEED)
    oof_preds = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in cv.split(X):
        fold_model = XGBRegressor(**params)
        fold_model.fit(X.iloc[train_idx][selected], y.iloc[train_idx])
        oof_preds[test_idx] = fold_model.predict(X.iloc[test_idx][selected])
    eval_oof = summarize_evaluation(y, oof_preds)

    # Refit final model
    final_model = XGBRegressor(**params)
    final_model.fit(X[selected], y)

    # Evaluate model
    y_pred = final_model.predict(X[selected])
    eval_train = summarize_evaluation(y, y_pred)
    fig_eval = plot_model_eval(np.exp(y), np.exp(y_pred))
    summary = [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.ListGroup(
                            [
                                dbc.ListGroupItem(f"Number of events: {len(sub)}"),
                                dbc.ListGroupItem(
                                    f"Number of cross-validation folds: {CV}"
                                ),
                                html.P(),
                            ]
                        ),
                        dbc.Row([dbc.Col(summ), dbc.Col(parm)]),
                        html.P(),
                        html.B("Model performance:"),
                        dbc.Row(
                            [
                                dbc.Col([html.I("CV (out-of-fold)"), eval_oof]),
                                dbc.Col([html.I("Refit (training)"), eval_train]),
                            ]
                        ),
                    ], md=6,
                ),
                dbc.Col(dcc.Graph(figure=fig_eval)),
            ]
        )
    ]

    # Feature importance plot
    importances = final_model.feature_importances_
    fig_feature = plot_feature_importance(importances, selected)

    # PDPs for each selected features
    figs_pdp = []
    if not selected:
        return (
            [html.P("No features were selected. Try different predictors.")],
            EMPTY_FIG,
            [],
        )
    for feat in selected:
        figs_pdp.append(plot_pdp(final_model, X[selected], feat, drivers))

    # Compute interaction strengths
    interaction_df = compute_interactions(final_model, X[selected], selected)
    fig_interaction = plot_interactions(interaction_df)

    return summary, fig_feature, figs_pdp, fig_interaction
