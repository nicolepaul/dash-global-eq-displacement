# layouts/layout_contributors.py
from dash import html
import dash_bootstrap_components as dbc


def layout_contributors():
    return dbc.Container(
        [
            html.P(
                "This research was made possible through the contribution of many experts around the world, including engineers, civil servants, and humanitarians. In some cases, these contributors provided housing damage or population displacement data, some of which was used directly and some of which was used to triangulate and verify data from other sources. In other cases, these contributors helped identify reliable national or local sources of data, facilitated connections with relevant contacts in civil protection or national ministries, offered qualitative contextual evidence related to displacement drivers, or assisted with translations between languages and damage scales. We gratefully acknowledge these contributions below."
            ),
            html.Div(id="ack-contents"),
        ],
        # fluid=True,
    )
