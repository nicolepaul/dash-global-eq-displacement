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
S = 5
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

NARRATIVE_REGRESSION = ""

NARRATIVE_DRIVERS = """This analysis fits machine learning models to predict the selected displacement metric 
                using a minimal number of predictors. Different environmental, economic, demographic, social, 
                and political drivers can be selected as explanatory variables."""

NARRATIVE_CORR = """This analysis is geared towards predictive models, which rely upon associations between different features 
                and the outcome variable. Associations or correlations are not sufficient to identify causality. Including 
                features that are highly correlated with one another can also lead to less stable model predictions. For example, 
                one might reasonably reduce features that have an absolute correlation coefficient above 80%."""
NARRATIVE_HIER = """This analysis uses correlations to group the explanatory drivers into clusters. For example, one might use this 
                analysis to assign natural clusters (e.g., income inequality, development level) and select one or two features in each."""
NARRATIVE_MI = """Mutual information captures the association strength between the explanatory variables and the selected displacement metric, 
                considering both linear and nonlinear relationships. For example, one might only select features ranking in the top half."""

NARRATIVE_FS = """To construct a practical predictive model, we seek to reduce any features that do not add meaningful predictive power. 
                In some cases, this will eliminate variables that have no clear relationship with the outcome variable, and in other cases 
                this might eliminate variables that are highly correlated with another variable already in the model."""
NARRATIVE_RFE = """To identify which limited set of mobility drivers best predict different displacement outcomes, recursive
                feature elimination (RFE) is performed. The RFE is run using a tree-based model (XGBoost), which 
                avoids assumptions about linearity and is robust to the inclusion of correlated features."""
NARRATIVE_FI = "A simple estimate of the feature importance for the selected variables is shown for the final XGBoost model."
NARRATIVE_PDP = """This analysis is ultimately intended to fit a simpler linear regression style model. The partial
                dependence plots help us understand whether the relationship between the predictors and the displacement 
                metric is linear, or whether some nonlinear terms require consideration."""
NARRATIVE_INT = """Interactions capture when the value of one feature influences the effect of another feature on model predictions. 
                The values here are calculated using the TreeExplainer from SHAP (SHapley Additive exPlanations). If certain features 
                exhibit a non-negligible interaction, then we might add interaction terms into our linear regression formulation."""


NARRATIVE_LIN = """A critical requirement for our probabilistic calculations is that our displacement model can interpolate and extrapolate. 
                Therefore, we fit linear regression models in this analysis. To accommodate potential nonlinear terms and interactions, we 
                investigate the results from tree-based models (XGBoost) to identify the top predictors and any relevant nonlinear relationships. 
                Once we've identified potential predictors, we fit linear regression models with every permutation of those predictors. To ensure 
                model stability, we repeat this process for many iterations of training/testing sample splits. Once we've identified the best 
                combination of predictors, we estimate the model uncertainty via bootstrapping."""

CORR_THRESH = 0.8
MI_QUANT = 0.5

PARAM_GRID = {
    "n_estimators": [50],
    "max_depth": [2, 3],
    "learning_rate": [0.1, 0.2, 0.3],
    "gamma": [0.3, 0.4, 0.5],
    "min_child_weight": [3, 4, 5, 6, 7],
}

PARAM_PROD = {
    "sheltered_peak": {
        "n_estimators": 50,
        "max_depth": 3,
        "learning_rate": 0.3,
        "gamma": 0.3,
        "min_child_weight": 4,
    },
    "protracted": {
        "n_estimators": 50,
        "max_depth": 2,
        "learning_rate": 0.3,
        "gamma": 0.5,
        "min_child_weight": 3,
    },
    "evacuated": {
        "n_estimators": 50,
        "max_depth": 2,
        "learning_rate": 0.2,
        "gamma": 0.5,
        "min_child_weight": 3,
    },
    "assisted": {
        "n_estimators": 50,
        "max_depth": 2,
        "learning_rate": 0.3,
        "gamma": 0.5,
        "min_child_weight": 4,
    },
}

LINEAR_TERMS = {
    "sheltered_peak": ["DESTROYED", "I(DESTROYED>10)", "INCOME", "I(INCOME>0.7)", "PALMA", "I(PALMA>0.065)", "TENURE_SECURITY", "I(TENURE_SECURITY>0.83)", "DESTROYED×INCOME"],
    "protracted": ["DESTROYED", "I(DESTROYED>10.3)", "PALMA", "I(PALMA>0.044)", "DESTROYED×PALMA"],
    "evacuated": ["DESTROYED","I(DESTROYED>9)", "I(DESTROYED>12.5)", "INCOME", "I(INCOME>0.69)"],
}
LINEAR_PROD = {
    "sheltered_peak": ['DESTROYED', 'I(INCOME>0.7)', 'I(PALMA>0.065)', 'DESTROYED×INCOME'],
    "protracted": ['DESTROYED', 'PALMA', 'I(PALMA>0.044)'],
    "evacuated": ['DESTROYED', 'I(DESTROYED>9)'],
}

CV_SEED = 22
MODEL_SEED = 99

LIN_REPEATS = 50
LIN_TEST = 0.2
N_BOOTSTRAP = 300

FILTER_R2 = 0.65
FILTER_MAPE = 0.45
FILTER_MDAPE = 0.20