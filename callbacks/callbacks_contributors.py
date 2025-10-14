from dash import Input, Output, html
import pandas as pd
from _config import *

def iso2_to_flag(iso2):
    if isinstance(iso2, str) and len(iso2) == 2:
        return ''.join(chr(127397 + ord(c.upper())) for c in iso2)
    return ""

def register_callbacks_contributors(app):
    @app.callback(
        Output("ack-contents", "children"),
        Input("main-tabs", "value"),
    )
    def render_ack(tab):
        if tab != "tab-contributors":
            return ""
        try:
            ack = pd.read_csv(PATH_ACK)
        except FileNotFoundError:
            return html.Div(f"{PATH_ACK} not found.")

        children = []
        regions = ack["region"].unique()
        sorted_regions = ["Global"] + sorted(r for r in regions if r != "Global")

        for region in sorted_regions:
            g = ack[ack["region"] == region]
            entries = []
            for _, row in g.iterrows():
                fields = [
                    str(row["name"]).strip(),
                    str(row["affiliation"]).strip() if pd.notna(row.get("affiliation")) and str(row["affiliation"]).strip() else None,
                ]
                country = str(row["country"]).strip() if pd.notna(row.get("country")) and str(row["country"]).strip() else None
                iso2 = str(row["iso2"]).strip() if pd.notna(row.get("iso2")) and str(row["iso2"]).strip() else None
                if country:
                    flag = iso2_to_flag(iso2)
                    if flag:
                        fields.append(f"{country} {flag}")
                    else:
                        fields.append(country)
                display = " — ".join([f for f in fields if f])
                entries.append(html.Div(display))
            children.append(html.Div([html.H5(region)] + entries))


        return children
