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

BORDERLESS = {"style": {"border": "none", "boxShadow": "none"}}


DEFAULT_TEXT = dbc.CardBody(
    [
        html.H4("Data definitions"),
        html.Ul(
            [
                html.Li(
                    [
                        html.B("Sheltered (peak): "),
                        "This is the number of people that seek collective shelter, or that required tents or shelter kits in more rural areas. This peak headcount is usually during the first two weeks after the mainshock, but sometimes will be in the first month. For high-income countries like the United States and Japan, the peak value was usually in the first few days. However, the peak value in low- and middle-income countries tended to occur later.",
                    ]
                ),
                html.Li(
                    [
                        html.B("Snapshot (X-month): "),
                        "This is the number of people who had persistent sheltering or housing needs after the earthquake. Estimates near the X month mark were prioritized, typically representing the population still in collective shelters or who were receiving some form of temporary or transitional housing from the government. Households receiving assistance are not included, as they often continued to occupy their homes. Additionally, households who independently found their own forms of temporary housing without humanitarian or government support are not included. ",
                    ]
                ),
                html.Li(
                    [
                        html.B("Damaged dwellings: "),
                        "This is the number of residential dwellings that have been damaged, but not destroyed. In general it is assumed that damaged buildings can be repaired. Often, households can occupy these dwellings during repair. An equivalent level of building loss would likely be <45%.",
                    ]
                ),
                html.Li(
                    [
                        html.B("Destroyed dwellings: "),
                        "This is the number of residential dwellings that have been destroyed. In general it is assumed that destroyed buildings cannot be easily repaired, and thus would likely be rebuilt new. At a minimum, these buildings are likely to be temporarily uninhabitable. An equivalent level of building loss would likely be somewhere between 45-65%.",
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
    "max_depth": [2],
    "learning_rate": [0.1, 0.2],
    "gamma": [0.1, 0.2, 0.3],
    "min_child_weight": [2, 3],
    "reg_alpha": [0.01, 0.1],
    "reg_lambda": [0.1, 1],
}

PARAM_PROD = { 
    "sheltered_peak": { # Consensus=15.2%, R2=0.802; DESTROYED, INCOME, PALMA, GRID_SMOD, AGE_DEPENDENCY, TENURE_SECURITY, EPR
        "n_estimators": 50,
        "max_depth": 2,
        "learning_rate": 0.1,
        "gamma": 0.2,
        "reg_alpha": 0.1,
        "reg_lambda": 1,
        "min_child_weight": 2,
    },
    "snapshot_3mo": { # Consensus=15.2%, R2=0.859; DESTROYED
        "n_estimators": 50,
        "max_depth": 2,
        "learning_rate": 0.1,
        "gamma": 0.3,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_child_weight": 2,
    },
    "snapshot_6mo": { # Consensus=13.1%, R2=0.844; DESTROYED, PALMA
        "n_estimators": 50,
        "max_depth": 2,
        "learning_rate": 0.2,
        "gamma": 0.2,
        "reg_alpha": 0.01,
        "reg_lambda": 0.1,
        "min_child_weight": 2,
    },
    "snapshot_12mo": { # Consensus=17.2%, R2=0.767; DESTROYED, PALMA
        "n_estimators": 50,
        "max_depth": 2,
        "learning_rate": 0.1,
        "gamma": 0.3,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_child_weight": 2,
    },
}

TREE_PROD = {
    "sheltered_peak": ["DESTROYED", "INCOME", "PALMA", "GRID_SMOD", "AGE_DEPENDENCY", "TENURE_SECURITY", "EPR"],
    "snapshot_3mo": ["DESTROYED"],
    "snapshot_6mo": ["DESTROYED", "PALMA"],
    "snapshot_12mo": ["DESTROYED", "PALMA"],
}

LINEAR_TERMS = {
    "sheltered_peak": ['DESTROYED', 'I(DESTROYED>5.7)', 'I(DESTROYED>8.2)', 'INCOME', 'I(INCOME>0.7)', 'PALMA', 'I(PALMA>0.06)', "GRID_SMOD", "AGE_DEPENDENCY", "I(AGE_DEPENDENCY>0.67)", "TENURE_SECURITY", "I(TENURE_SECURITY>0.81)", "EPR",'DESTROYED×INCOME'], 
    "snapshot_3mo": ["DESTROYED", "I(DESTROYED>5.5)", "I(DESTROYED>6.5)", "I(DESTROYED>12.6)"],
    "snapshot_6mo": ["DESTROYED", "I(DESTROYED>6.5)", "PALMA", "I(PALMA>0.044)", "I(PALMA>0.07)","DESTROYED×PALMA"],
    "snapshot_12mo": ["DESTROYED","I(DESTROYED>5.5)","I(DESTROYED>8.9)", "PALMA", "I(PALMA>0.044)", "I(PALMA>0.07)", "DESTROYED×PALMA"],
}

LINEAR_PROD = {
    "sheltered_peak": ['DESTROYED', 'I(DESTROYED>5.7)', 'I(INCOME>0.7)', 'I(PALMA>0.06)', 'I(AGE_DEPENDENCY>0.67)'], # Held-out splits: MdAPE = 11.2%, MAPE = 15.3%, R² = 79.8%, Consensus score = 15.6%, Sigma = 1.642
    "snapshot_3mo": ['DESTROYED'], # Held-out splits: MdAPE = 10.4%, MAPE = 12.3%, R² = 83.8%, Consensus score = 13.0%, Sigma = 1.973
    "snapshot_6mo": ['DESTROYED', 'PALMA', 'DESTROYED×PALMA'], # Held-out splits: MdAPE = 10.2%, MAPE = 11.2%, R² = 84.1%, Consensus score = 12.4%, Sigma = 1.808
    "snapshot_12mo": ['DESTROYED', 'PALMA', 'DESTROYED×PALMA'], # Held-out splits: MdAPE = 8.6%, MAPE = 11.4%, R² = 84.3%, Consensus score = 11.9%, Sigma = 1.936
}

CV_SEED = 22
MODEL_SEED = 99

CV = 5
S = 10
MIN_EVENT = 20

MAX_TERMS = 5
LIN_REPEATS = 50
LIN_TEST = 0.2
N_BOOTSTRAP = 300