import os
import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.graph_objs as go

PATH_DATA = os.path.join("assets", "data.csv")
PATH_DRIVERS = os.path.join("assets", "drivers.csv")
PATH_ACK = os.path.join("assets", "acknowledgments.csv")

CATEGORICAL_COLORS = [
    "#44aa98",
    "#ab4498",
    "#332389",
    "#86ccec",
    "#ddcc76",
    "#cd6477",
    "#882255",
    "#117732",
    "#666666",
    "#212121",
]

FILL_ZERO = 0.4

EMPTY_FIG = go.Figure()
EMPTY_FIG.update_layout(
    xaxis={"visible": False},
    yaxis={"visible": False},
    annotations=[
        {
            "text": "No data to display",
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {"size": 18},
        }
    ],
    height=300,
    margin=dict(l=0, r=0, t=40, b=0),
)

CV = 5
S = 3
MIN_EVENT = 20

BORDERLESS = {"style": {"border": "none", "boxShadow": "none"}}


DEFAULT_TEXT = dbc.CardBody(
    [
        html.H4("Data definitions"),
        html.Ul(
            [
                html.Li(
                    [
                        html.B("Evacuated (peak): "),
                        "This is the number of people who leave their habitual dwelling for any period of time, whether that is to stay with family/friends, sleep outdoors on their land, or to go to a collective shelter point. This data is rarely systematically captured, although some countries such as the Philippines regularly track displaced populations both in collective shelters and with host families.",
                    ]
                ),
                html.Li(
                    [
                        html.B("Sheltered (peak): "),
                        "This is the number of people that seek collective shelter, or that required tents or shelter kits in more rural areas. This peak headcount is usually during the first two weeks after the mainshock, but sometimes will be in the first month.",
                    ]
                ),
                html.Li(
                    [
                        html.B("Protracted (6-month): "),
                        "This is the number of people who had persistent sheltering or housing needs after the earthquake. Estimates near the six month mark were prioritized, typically representing the population still in collective shelters or who were receiving some form of temporary or transitional housing from the government.",
                    ]
                ),
                html.Li(
                    [
                        html.B("Assisted: "),
                        "This is the number of people who received some form sheltering or housing assistance after the earthquake. Forms of assistance could include temporary or transitional housing, cash assistance for repairs or rebuilding, rental voucher, or replacement housing. Often, governments distribute this form of assistance based on housing damage, resulting in a high correlation between the reported damage and the reported number of people receiving assistance. Monetary assistance is the most prevalent form, but amounts received are not necessarily sufficient to cover costs to repair or rebuild.",
                    ]
                ),
            ]
        ),
    ]
)

NARRATIVE_REGRESSION = (
            html.P(),
            "Supported linear regression models:",
            html.Ul(
                [
                    html.Li("OLS: Ordinary least squares"),
                    html.Li("RLM: Robust linear model"),
                ]
            ),
        )

NARRATIVE_DRIVERS = """This analysis fits machine learning models to predict the selected displacement metric 
                using a minimal number of predictors. Different environmental, economic, political, social, 
                and demographic displacement drivers can be selected as explanatory variables."""

NARRATIVE_CORR = """This analysis is geared towards predictive models, which rely upon associations between different features 
                and the outcome variable. Associations or correlations are not sufficient to identify causality. Including 
                features that are highly correlated with one another can also lead to less stable model predictions."""

NARRATIVE_FS = """To construct a practical predictive model, we seek to reduce any features that do not add meaningful predictive power. 
                In some cases, this will eliminate variables that have no clear relationship with the outcome variable, and in other cases 
                this might eliminate variables that are highly correlated with another variable already in the model."""
NARRATIVE_RFE = """To identify which limited set of mobility drivers best predict different displacement outcomes, recursive
                feature elimination (RFE) is performed. The RFE is run using a tree-based model (XGBoost), which 
                avoids assumptions about linearity and is robust to the inclusion of correlated features."""
NARRATIVE_FI = "A simple estimate of the feature importance for the selected variables is shown for the final XGBoost model."
NARRATIVE_PDP = """This analysis is ultimately intended to fit a simpler linear regression style model. The partial
                dependence plots help us understand whether the relationship between the predictors and the displacement                 metric is linear, or whether some nonlinear terms require consideration."""
