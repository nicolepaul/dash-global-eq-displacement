import os
import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.graph_objs as go

PATH_DATA = os.path.join("assets", "data.csv")
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