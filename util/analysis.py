import numpy as np
import statsmodels.formula.api as smf
import plotly.graph_objs as go
from dash import dcc, html
import dash_bootstrap_components as dbc
from _config import *


def transform_variables(df, drivers):
    for i, row in drivers.iterrows():
        if row['transform'] == 'log':
            df[row.variable] = np.log1p(df[row.variable])
            drivers.loc[i, 'name'] = 'log(' + drivers.loc[i, 'name'] + ')'
    return df, drivers

def run_regression(df, x_col, y_col, method="ols", add_trace=True):

    N = 200

    df["log_x"] = np.log1p(df[x_col])
    df["log_y"] = np.log1p(df[y_col])

    eqn = None
    lm_fit = None
    color, style = 'black', 'dash'
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
        color, style = 'darkred', 'dot'
    elif method == "rlm_int":
        lm = smf.rlm(formula="log_y ~ log_x", data=df)
        lm_fit = lm.fit()
        eqn = html.Div(
                    [
                        dcc.Markdown("$\\log y=\\beta \\cdot \\log x + \\alpha$", mathjax=True),
                        dcc.Markdown("Or, $y = e^{\\alpha}x^{\\beta}$", mathjax=True),
                    ]
                )
        color, style = 'darkred', 'dot'
    else:
        raise NotImplementedError(f"Method '{method}' not supported yet.")

    # Get key values
    alpha = lm_fit.params["Intercept"] if hasattr(lm_fit.params, "Intercept") else None
    beta = lm_fit.params["log_x"] if hasattr(lm_fit.params, "log_x") else None
    r2 = lm_fit.rsquared if hasattr(lm_fit, "rsquared") else None

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
        log_x = np.linspace(np.log1p(FILL_ZERO), df['log_x'].max(), N)
        y_hat = lm_fit.predict(exog=dict(log_x=log_x))

        trace = go.Scatter(
            x=np.expm1(log_x),
            y=np.expm1(y_hat),
            mode="lines",
            name=f"{method.upper()} (β={beta:.3f})",
            line=dict(color=color, dash=style),
            hoverinfo='skip',
            zorder=5,
        )

    return trace, narrative

def mape_from_log(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mask = y_true != 0
    return np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))