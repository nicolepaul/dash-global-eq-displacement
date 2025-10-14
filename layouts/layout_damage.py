from dash import dcc, html
import dash_bootstrap_components as dbc
from _config import *

def scatter_graph():
    return dbc.Card(
        [
            # html.H4("Earthquake event data"),
            html.P(),
            dbc.Row(dcc.Graph(id="scatter-graph")),
            dbc.Row(
                html.Em(
                    "Note: Zero values were replaced with 0.4 to display on a log-log plot "
                    "and perform calculations in logspace"
                )
            ),
        ],
        body=True,
    )


def regression_section():
    return dbc.Card(
        [
            dbc.Row(
                [
                    html.P(),
                    html.H4("First-order analysis"),
                    html.P(
                        "We can fit a simple linear regression model to approximate displacement based on damage:"
                    ),
                ]
            ),
            dbc.Row(
                [
                    dcc.RadioItems(
                        id="regression-radio",
                        options=[
                            {"label": "No regression", "value": "none"},
                            {"label": "Run OLS without intercept", "value": "ols"},
                            {"label": "Run OLS with intercept", "value": "ols_int"},
                        ],
                        value="none",  # Default selection
                        inline=False,
                        inputStyle={"margin-right": "8px", "margin-left": "8px"},
                    ),
                ]
            ),
            dbc.Row(
                html.Div(
                    id="regression-narrative",
                )
            ),
        ],
        body=True,
    )


def controls(xs, ys, default_x, default_y):
    return dbc.Card(
        [
            html.H4("Select variables"),
            html.B("Displacement metric, y"),
            dcc.Dropdown(
                id="y-selector",
                options=[{"label": ys[name], "value": name} for name in ys],
                value=default_y,
                clearable=False,
                searchable=True,
            ),
            html.B("Damage estimate, x"),
            dcc.Dropdown(
                id="x-selector",
                options=[{"label": xs[name], "value": name} for name in xs],
                value=default_x,
                clearable=False,
                searchable=True,
            ),
        ],
        body=True,
    )

def event_narrative():
    return dbc.Card(
        [
            dcc.Store(id="narrative-mode", data="default"),
            dcc.Store(id="narrative-event-data", data=None),
            dbc.Card(id="event-narrative", body=True)
        ],
        body=True
    )


def layout_damage(xs, ys):
    # Default x/y for initial view
    default_y = "sheltered_peak"
    default_x = "destroyed*ahhs-fatalities"

    return dbc.Container(
        [
            # stores(),
            dbc.Row(
                [
                    dbc.Col(
                        [dbc.Row(scatter_graph()), dbc.Row(event_narrative())], md=8
                    ),
                    dbc.Col(
                        [
                            controls(xs, ys, default_x, default_y),
                            regression_section(),
                        ],
                        md=4,
                    ),
                ],
            ),
        ],
        fluid=True,
    )
