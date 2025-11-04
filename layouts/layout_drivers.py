from dash import dcc, html
import dash_bootstrap_components as dbc
from _config import *

# TODO: We should include the number of events with selected drivers somewhere to help guide decisions;
# have some visual indication of model performance


def control_analysis():
    return dbc.Row(
        [
            html.H4("Run analysis"),
            html.H5('Initial feature selection'),
            html.P('Before fitting models, we need to reduce the number of explanatory variables to avoid multi-colinearity and overfitting.'),
            dbc.Button(
                "Explore drivers",
                id="explore-btn",
                style={
                    "width": "40%",
                    "margin": "0.5rem",
                },
                n_clicks=0,
            ),
            html.H5('Model fitting'),
            html.P('Tree-based models are useful for reducing the number of predictors and understanding the nature of their relationship with the selected displacement metric. Once the best predictors are identified, nonlinear terms can be incorporated into linear regression-style models for practical implementation.'),
            dcc.Dropdown(
                id="model-type-dropdown",
                options=[{"label": "Tree-based (XGBoost)", "value": "xgboost"}, {"label": "Linear regression", "value": "linear"}],
                value="linear",
                placeholder="Select model type",
                style={"marginBottom": "0.25em"},
            ),
            dbc.Button(
                "Run analysis",
                id="analysis-btn",
                style={
                    "width": "40%",
                    "margin": "0.5rem",
                },
                n_clicks=0,
            ),
            html.P(),
            html.Hr(),
        ]
    )


def select_metric(ys):

    return dbc.Row(
        [
            html.H4('Select variables'),
            html.H5("Displacement metric"),
            dcc.Dropdown(
                id="rfe-metric-dropdown",
                options=[{"label": ys[name], "value": name} for name in ys if name != 'assisted'],
                value="sheltered_peak",
                placeholder="Select displacement metric",
                style={"marginBottom": "1em"},
            ),
        ]
    )


def select_variables(drivers):
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
    return dbc.Row(
        [html.H5("Explanatory variables"), html.Div(checklist_groups), html.Br()]
    )


def analysis_narrative():
    return dbc.Col(
        NARRATIVE_DRIVERS,
        style={"marginBottom": "1em"},
    )


def layout_drivers(ys, drivers):
    return dbc.Container(
        [
            html.P(),
            dbc.Row(analysis_narrative()),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            control_analysis(),
                            select_metric(ys),
                            select_variables(drivers),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        dcc.Loading(
                            id="loading-drivers-results",
                            children=html.Div(id="drivers-results-container"),
                        )
                    ),
                ]
            ),
        ],
        fluid=True,
    )
