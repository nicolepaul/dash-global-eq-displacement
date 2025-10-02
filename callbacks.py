import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, dcc, callback_context

from _config import *
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
                html.Div(
                    dcc.Markdown("$\\log y=\\alpha \\cdot \\log x$", mathjax=True)
                ),
                # html.Div(dcc.Markdown("Or, equivalently: $y = x^\\alpha$", mathjax=True)),
                html.B("Regression Results"),
                regression_results,
            ]

        return go.Figure(data=traces, layout=layout), narrative

    @app.callback(
        Output("event-narrative", "children"),
        Input("scatter-graph", "clickData"),
        Input("reset-event-narrative", "n_clicks"),
        prevent_initial_call=True,
    )
    def update_event_narrative(clickData, reset_n_clicks):
        ctx = callback_context  

        # If reset link was clicked, show default text
        if ctx.triggered and ctx.triggered[0]["prop_id"].startswith("reset-event-narrative"):
            return DEFAULT_TEXT

        # Otherwise, handle clickData
        def parse_event_info(row):
            heading = html.H4(row["event"] + ", " + row["country"])
            body = html.P("A narrative about this event will eventually go here")
            footing = html.A(
                "Return to the global data definitions →",
                href="#",
                id="reset-event-narrative",
            )
            return dbc.CardBody([heading, body, footing])

        if clickData and "points" in clickData and len(clickData["points"]) > 0:
            point = clickData["points"][0]
            row = point.get("customdata", None)
            if row:
                return parse_event_info(row)

        return DEFAULT_TEXT