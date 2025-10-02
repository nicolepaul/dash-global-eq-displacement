from dash import dcc, html
import dash_bootstrap_components as dbc

BORDERLESS = {"style": {"border": "none", "boxShadow": "none"}}


def header():
    return dbc.Card(
        [
            html.H1("🌏🌎🌍 Population displacement after earthquakes"),
            html.P(
                """
                This dashboard visualizes newly assembled data on population displacement
                after recent earthquakes around the world, as well as potential
                displacement drivers such as housing damage.
                Research in progress, with support from the UCL Overseas Research Scholarship,
                WTW Research Network, and IDMC.
                """
            ),
            html.Em(
                [
                    "Dashboard created by ",
                    html.A("Nicole Paul", href="https://nicolepaul.io"),
                ]
            ),
        ],
        body=True,
        **BORDERLESS,
    )


def footer():
    return dbc.Card([], body=True, **BORDERLESS)


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


def scatter_graph():
    return dbc.Card(
        [
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
                        "We can fit a simple linear regression model to approximate displacement based on damage."
                    ),
                ]
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Checkbox(
                        id="regression-check",
                        value=False,
                        className="form-check-input",
                        label="Fit OLS regression",
                        style={
                            "whiteSpace": "nowrap",
                            "width": "100%",
                            "border": "0px",
                        },
                    ),
                )
            ),
            dbc.Row(
                html.Div(
                    id="regression-narrative",
                    **BORDERLESS,
                )
            ),
        ],
        body=True,
        **BORDERLESS,
    )


def create_layout(xs, ys, df):
    # Default x/y for initial view
    default_y = "sheltered_peak"
    default_x = "destroyed*ahhs-fatalities"

    return dbc.Container(
        [
            dbc.Row([header()]),
            dbc.Row(
                [
                    dbc.Col(scatter_graph(), md=8),
                    dbc.Col(
                        [controls(xs, ys, default_x, default_y), regression_section()],
                        md=4,
                    ),
                ],
            ),
            dbc.Row([footer()]),
        ],
        fluid=True,
    )
