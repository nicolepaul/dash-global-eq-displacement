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
                            {"label": "Ordinary least squares, with intercept", "value": "ols_int"},
                            {"label": "Ordinary least squares, without intercept", "value": "ols"},
                            {"label": "Robust linear model, with intercept", "value": "rlm_int"},
                            {"label": "Robust linear model, without intercept", "value": "rlm"},
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


def controls(xs, ys, zs, default_x, default_y, default_z):
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
            html.B("Color data by"),
            dcc.Dropdown(
                id="z-selector",
                options=[{"label": zs[name], "value": name} for name in zs],
                value=default_z,
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
        body=True,
    )


def layout_damage(xs, ys, zs):
    # Default x/y/z for initial view
    default_y = "sheltered_peak"
    default_x = "destroyed*ahhs-fatalities"
    default_z = "region"

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
                            controls(xs, ys, zs, default_x, default_y, default_z),
                            regression_section(),
                        ],
                        md=4,
                    ),
                ],
            ),
        ],
        fluid=True,
    )
