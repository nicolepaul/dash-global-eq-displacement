import dash_bootstrap_components as dbc
from dash import dcc, html

BORDERLESS = {"style": {"border": "none", "boxShadow": "none"}}


DEFAULT_TEXT = dbc.CardBody(
    [
        html.H4("Data definitions"),
        html.Ul(
            [
                html.Li(
                    [
                        html.B("Evacuated: "),
                        "This is the number of people who leave their habitual dwelling for any period of time, whether that is to stay with family/friends, sleep outdoors on their land, or to go to a collective shelter point. This data is rarely systematically captured, although some countries such as the Philippines regularly track displaced populations both in collective shelters and with host families.",
                    ]
                ),
                html.Li(
                    [
                        html.B("Sheltered: "),
                        "This is the number of people that seek collective shelter, or that required tents or shelter kits in more rural areas. We represent this data at different time periods, with the most commonly reported estimate reflecting a peak shelter headcount. This peak headcount is usually during the first two weeks after the mainshock, but sometimes will be in the first month.",
                    ]
                ),
                html.Li(
                    [
                        html.B("Needs: "),
                        "This is the number of people who required sheltering or housing assistance after the earthquake. Estimates near the six month mark were prioritized, typically representing the population still in collective shelters or who were receiving some form of temporary or transitional housing from the government.",
                    ]
                ),
                html.Li(
                    [
                        html.B("Assisted: "),
                        "This is the number of people who rreceived sheltering or housing assistance after the earthquake. Forms of assistance could include temporary or transitional housing, cash assistance for repairs or rebuilding, rental voucher, or replacement housing. Often, governments distribute this form of assistance based on housing damage, resulting in a high correlation between the reported damage and the reported number of people receiving assistance.",
                    ]
                ),
            ]
        ),
    ]
)