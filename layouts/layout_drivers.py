from dash import dcc, html
import dash_bootstrap_components as dbc


def select_drivers(xs, ys, df):
    return dbc.Col(
                    [
                        html.Label("Displacement metric"),
                        dcc.Dropdown(
                            id="rfe-metric-dropdown",
                            options=[{"label": ys[name], "value": name} for name in ys],
                            placeholder="Select displacement metric",
                        ),
                        html.Br(),
                        html.Label("Candidate predictors"),
                        dcc.Dropdown(
                            id="rfe-predictors-dropdown",
                            options=[{"label": xs[name], "value": name} for name in xs]
                            + [
                                {"label": c, "value": c}
                                for c in df.columns
                                if c == c.upper()
                            ],
                            multi=True,
                            placeholder="Select predictors",
                        ),
                        html.Br(),
                        dbc.Button("Run analysis", id="rfe-run-btn"),
                    ],
                    width=4,
                )

def rfe_summary():
    return dbc.Col(
                    [
                        html.H5("Results summary"),
                        html.Div(id="rfe-summary"),
                        html.Br(),
                        dcc.Graph(
                            id="rfe-feature-importance", style={"height": "300px"}
                        ),
                    ],
                    width=8,
                )

def pdp_plots():
    return dbc.Col(
        [
            html.H5("Partial dependence plots"),
            html.Div(id="rfe-pdp-container")
        ]
    )


def layout_drivers(xs, ys, df):
    return dbc.Container(
            [
                html.P("Coming soon!"),
            ],
            fluid=True,
        )


# def layout_drivers(xs, ys, df):
#     return dbc.Container(
#             [
#                 html.P(),
#                 dbc.Row([select_drivers(xs, ys, df), rfe_summary()]),
#                 html.Hr(),
#                 pdp_plots(),
#             ],
#             fluid=True,
#         )
