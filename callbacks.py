import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, dcc

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
        Output("narrative-mode", "data"),
        Output("narrative-event-data", "data"),
        Input("scatter-graph", "clickData"),
        prevent_initial_call=True,
    )
    def set_event_mode(clickData):

        if clickData is None:
            return "default", None
        point = clickData["points"][0]
        return "event", point.get("customdata", None)

    # When back link is clicked, reset to default
    @app.callback(
        Output("narrative-mode", "data", allow_duplicate=True),
        Output("narrative-event-data", "data", allow_duplicate=True),
        Input("back-link", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_mode(_):
        return "default", None

    # Main narrative renderer
    @app.callback(
        Output("event-narrative", "children"),
        Input("narrative-mode", "data"),
        State("narrative-event-data", "data"),
    )
    def render_narrative(mode, event_data):

        def parse_event_info(row):
            heading = html.H4(row["event"] + ", " + row["country"])
            body = html.P("A narrative about this event will eventually go here")
            footing = html.A("← Back to global data definitions", href="#", id="back-link")
            return dbc.CardBody([heading, body, footing])
        
        if mode == "default":
            return dbc.CardBody(DEFAULT_TEXT)

        if mode == "event" and event_data is not None:
            return parse_event_info(event_data)

        return dbc.CardBody(DEFAULT_TEXT)