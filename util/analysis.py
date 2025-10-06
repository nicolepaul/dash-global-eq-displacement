import numpy as np
import statsmodels.formula.api as smf
import plotly.graph_objs as go
from dash import dcc, html
import dash_bootstrap_components as dbc

def run_regression(df, x_col, y_col, method="ols", add_trace=True):

    N = 200
    nar = None

    df["log_x"] = np.log10(df[x_col])
    df["log_y"] = np.log10(df[y_col])

    eqn = None
    lm_fit = None
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
        # Generate prediction grid
        x_vals = np.logspace(df["log_x"].min(), df["log_x"].max(), N, base=10)
        log_x = np.log10(x_vals)
        y_hat = lm_fit.predict(exog=dict(log_x=log_x))

        trace = go.Scatter(
            x=x_vals,
            y=10 ** y_hat,
            mode="lines",
            name=f"{method.upper()} (β={beta:.3f})",
            line=dict(color="black", dash="dash"),
            hoverinfo='skip',
            zorder=5,
        )

    return trace, narrative
