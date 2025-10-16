from dash import dcc, html
import dash_bootstrap_components as dbc


def select_drivers(xs, ys, df, drivers):
    categories = drivers["category"].unique()
    checklist_groups = []

    for cat in categories:
        group = drivers[drivers["category"] == cat]
        checklist_items = []
        for _, row in group.iterrows():
            checklist_items.append(
                html.Div(
                    [
                        dbc.Checkbox(
                            id={"type": "driver-checkbox", "index": row["variable"]},
                            value=bool(row.get("default", False)),
                            style={"display": "inline-block"},
                        ),
                        html.Span(
                            row["name"],
                            style={"display": "inline-block"},
                        ),
                    ],
                )
            )

        checklist_groups.append(
            html.Div(
                [
                    html.B(cat),
                    html.Div(checklist_items),
                ],
                style={"marginBottom": "1em"},
            )
        )

    return dbc.Col(
        [
            html.H4("Displacement metric"),
            dcc.Dropdown(
                id="rfe-metric-dropdown",
                options=[{"label": ys[name], "value": name} for name in ys],
                value="sheltered_peak",
                placeholder="Select displacement metric",
                style={"marginBottom": "1em"},
            ),
            dbc.Button(
                "Run analysis", id="analysis-btn", className="load-button", n_clicks=0
            ),
            html.Hr(),
            html.H4("Explanatory variables"),
            html.Div(checklist_groups),
            html.Br(),
        ],
        width=4,
    )


def analysis_narrative():
    return dbc.Col(
        [
            html.P(
                """This analysis fits machine learning models to predict the selected displacement metric 
                using a minimal number of predictors. Different environmental, economic, political, social, 
                and demographic displacement drivers can be selected as explanatory variables."""
            ),
            html.P(
                """This analysis is geared towards predictive models, which rely upon associations between different features 
                and the outcome variable. Associations or correlations are not sufficient to identify causality. 
                To construct a practical predictive model, we seek to reduce any features that do not add meaningful predictive power. 
                In some cases, this will eliminate variables that have no clear relationship with the outcome variable, and in other cases 
                this might eliminate variables that are highly correlated with another variable already in the model."""
            ),
        ],
        style={"marginBottom": "1em"},
    )


# def rfe_summary():
#     return dbc.Col(
#         [
#             html.H4("Recursive feature elimination"),
#             html.P(
#                 """To identify which limited set of mobility drivers best predict different displacement outcomes, recursive
#                 feature elimination (RFE) is performed. The RFE is run using a tree-based model (XGBoost), which 
#                 avoids assumptions about linearity and is robust to the inclusion of correlated features."""
#             ),
#             html.Div(id="rfe-summary"),
#         ],
#     )


# def rfe_importance():
#     return dbc.Col(
#         [
#             html.H4("Feature importance"),
#             html.P(
#                 "A simple estimate of the feature importance for the selected variables is shown for the final XGBoost model."
#             ),
#             html.Div(id="rfe-feature-importance"),
#         ],
#     )


# def pdp_plots():
#     return dbc.Col(
#         [
#             html.H4("Partial dependence plots"),
#             html.P(
#                 """This analysis is ultimately intended to fit a simpler linear regression style model. The partial
#                 dependence plots help us understand whether the relationship between the predictors and the displacement 
#                 metric is linear, or whether some nonlinear terms require consideration."""
#             ),
#             html.Div(id="rfe-pdp-container"),
#         ]
#     )


# def corr_summary():
#     return dbc.Col(
#         [
#             html.H4("Correlation analysis"),
#             html.P(
#                 "Linear regression models are not robust to correlated features, so we should be careful about including them."
#             ),
#             html.Div(id="corr-container"),
#         ]
#     )


# def layout_drivers(xs, ys, df):
#     return dbc.Container(
#             [
#                 html.P("Coming soon!"),
#             ],
#             fluid=True,
#         )


def layout_drivers(xs, ys, df, drivers):
    return dbc.Container(
        [
            html.P(),
            dbc.Row(analysis_narrative()),
            dbc.Row(
                [
                    select_drivers(xs, ys, df, drivers),
                    dbc.Col(
                        dcc.Loading(
                            id="loading-drivers-results",
                            type="default",
                            children=html.Div(id="drivers-results-container"),
                        )
                    ),
                ]
            ),
        ],
        fluid=True,
    )