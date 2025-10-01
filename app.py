import os

import dash
import plotly.graph_objs as go
from dash import Input, Output, dcc, html
from dash_bootstrap_templates import load_figure_template
import dash_bootstrap_components as dbc

from util.parsers import get_data
from util.plotters import arrange_scatter
from util.analysis import run_regression

# Retrieve data and initial inputs
df, xs, ys = get_data()
y = "sheltered_peak"
x = "destroyed*ahhs-fatalities"

# Initialize app
app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
load_figure_template("FLATLY")
app.title = "Population displacement after earthquakes"

# Header content
header_content = [
    html.H1("🌏🌎🌍 Population displacement after earthquakes"),
    html.P(
        """
            This dashboard visualizes newly assembled data on population displacement
            after recent earthquakes around the world, as well as different potential
            displacement drivers such as housing damage.
            This research is still in progress, with support from the UCL Overseas Research Scholarship, WTW Research Newtork, and IDMC.
        """
    ),
    html.Em(
        [
            "Dashboard created by ",
            html.A("Nicole Paul", href="https://nicolepaul.io"),
        ]
    ),
]

# Footer content
footer_content = []

# Create controls
control_y = html.Div(
    [
        html.B("Displacement metric, y"),
        dcc.Dropdown(
            id="y-selector",
            options=[
                {
                    "label": ys[name],
                    "value": name,
                }
                for name in ys
            ],
            value=y,
            clearable=False,
            searchable=True,
        ),
    ]
)
control_x = html.Div(
    [
        html.B("Damage estimate, x"),
        dcc.Dropdown(
            id="x-selector",
            options=[
                {
                    "label": xs[name],
                    "value": name,
                }
                for name in xs
            ],
            value=x,
            clearable=False,
            searchable=True,
        ),
    ]
)
control_regression = html.Div(
    dbc.Checkbox(
        id="regression-check",
        value=False,
        className="form-check-input",
        label="Fit OLS regression",
        style={"whiteSpace": "nowrap", "width": "100%", "border": "0px"},
    )
)

# Create graphs
graph_controls = dbc.Card([html.H4("Select variables"), control_y, control_x], body=True)
graph_scatter = dbc.Card(
    [
        dbc.Row(dcc.Graph(id="scatter-graph")),
        dbc.Row(
            html.Em(
                "Note: Zero values were replaced with 0.4 to display on a log-log plot and perform calculations in logspace"
            )
        ),
    ],
    body=True,
)

# Narrative
narrative_header = [html.P(), html.H4("Regression analysis"), control_regression]
narrative_scatter = dbc.Card(
    id="narrative-scatter", body=True, style={"border": "none", "boxShadow": "none"}
)
narrative_regression = dbc.Card(
    id="narrative-regression",
    body=True,
    style={"border": "none", "boxShadow": "none"},
)
narrative = dbc.Card(
    [dbc.Row(narrative_header), dbc.Row(narrative_scatter), dbc.Row(narrative_regression)],
    style={"border": "none", "boxShadow": "none"},
)

# Store results
dcc.Store(id="regression-results")

# Create layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Card(
                    header_content,
                    body=True,
                    style={"border": "none", "boxShadow": "none"},
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(graph_scatter, md=8),
                dbc.Col([dbc.Row(graph_controls), dbc.Row(narrative)], md=4),
            ],
        ),
        dbc.Row(
            [
                dbc.Card(
                    footer_content,
                    body=True,
                    style={"border": "none", "boxShadow": "none"},
                )
            ]
        ),
    ],
    fluid=True,
)

@app.callback(
    Output("scatter-graph", "figure"),
    Output("narrative-regression", "children"),
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
        narrative = [
            html.B("Regression Results"),
            html.Plaintext(f"α = {alpha:.2f}"),
            html.Plaintext(f"R² = {r2:.3f}" if r2 is not None else "R² not available"),
        ]

    return go.Figure(data=traces, layout=layout), narrative


@app.callback(
    Output("narrative-scatter", "children"),
    [Input("y-selector", "value"), Input("x-selector", "value")],
)
def narrative_scatter(y_choice, x_choice):
    return [
        html.P(
            f"Fitting a regression model using data from {df[y_choice].count():,.0f} recent earthquake events globally:"
        ),
        html.Div(dcc.Markdown("$\\log y=\\alpha \\cdot \\log x$", mathjax=True)),
        html.Div(dcc.Markdown("Or, equivalently: $y = x^\\alpha$", mathjax=True)),
    ]

if __name__ == "__main__":
    app.run(port=os.environ.get('PORT', 10000))
    server = app.server
