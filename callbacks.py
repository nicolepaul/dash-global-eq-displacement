import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from dash import Input, Output, html, dcc

from util.analysis import run_regression
from util.plotters import arrange_scatter


def register_callbacks(app, df):

    @app.callback(
        Output("scatter-graph", "figure"),
        Output("regression-narrative", "children"),
        Input("y-selector", "value"),
        Input("x-selector", "value"),
        Input("regression-check", "value"),
    )
    def update_outputs(y_choice, x_choice, regression):
        df_fit = df.copy()
        traces, layout = arrange_scatter(df_fit, y_choice, x_choice)

        alpha, r2, trace = None, None, None
        if regression:
            alpha, r2, trace = run_regression(df, x_choice, y_choice, method="ols")
            if trace:
                traces.append(trace)

        narrative = []
        if alpha is not None:

            regression_results = dbc.ListGroup(
                [
                    dbc.ListGroupItem(f"n = {df[y_choice].count():,.0f}"),
                    dbc.ListGroupItem(f"α = {alpha:.3f}"),
                    dbc.ListGroupItem(
                        f"R² = {r2:.3f}" if r2 is not None else "R² not available"
                    ),
                ],
                flush=True,
            )
            narrative = [
                html.P(),
                html.Div(dcc.Markdown("$\\log y=\\alpha \\cdot \\log x$", mathjax=True)),
                # html.Div(dcc.Markdown("Or, equivalently: $y = x^\\alpha$", mathjax=True)),
                html.B("Regression Results"),
                regression_results,
            ]

        return go.Figure(data=traces, layout=layout), narrative
