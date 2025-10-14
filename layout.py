from dash import dcc, html
import dash_bootstrap_components as dbc

from _config import *

from layouts.layout_damage import layout_damage
from layouts.layout_drivers import layout_drivers
from layouts.layout_contributors import layout_contributors


def header():
    return dbc.Card(
        [
            html.H1("Population displacement after earthquakes 🌏🌎🌍"),
            html.P(
                """
                This dashboard visualizes newly assembled data on population displacement
                after recent earthquakes around the world, as well as potential
                displacement drivers such as housing damage.
                Research is still in progress, and this dashboard is not yet intended for a general audience.
                """
            ),
        ],
        body=True,
    )


def footer():
    return dbc.Card(
        [
            html.P(
                "This research was financially supported by the Internal Displacement Monitoring Centre (IDMC), University College London (UCL), and the Willis Towers Watson (WTW) Research Network"
            ),
            html.Em(
                [
                    "Dashboard created by ",
                    html.A(
                        "Nicole Paul", href="https://nicolepaul.io", target="_blank"
                    ),
                ]
            ),
        ],
        body=True,
    )


def main_layout(xs, ys, df):
    return dbc.Container(
        [
            dbc.Row([header()]),
            dbc.Row(
                [
                    dcc.Tabs(
                        id="main-tabs",
                        value="tab-damage",
                        children=[
                            dcc.Tab(label="Event data", value="tab-damage"),
                            dcc.Tab(label="Displacement drivers", value="tab-drivers"),
                            dcc.Tab(label="Acknowledgments", value="tab-contributors"),
                        ],
                    )
                ]
            ),
            dbc.Row([html.Div(id="tab-content")]),
            dbc.Row([footer()]),
        ],
        fluid=True,
    )
