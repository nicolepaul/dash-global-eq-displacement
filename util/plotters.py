import numpy as np
import plotly.graph_objs as go

from _config import *


def arrange_scatter(df, y_choice, x_choice):

    regions = df["region"].unique()
    REGION_COLORS = {regions[i]: CATEGORICAL_COLORS[i] for i in range(len(regions))}

    df["log_x"] = np.log10(df[x_choice])
    df["log_y"] = np.log10(df[y_choice])

    traces = []
    for region in regions:
        sub = df[df["region"] == region]
        traces.append(
            go.Scatter(
                x=sub[x_choice],
                y=sub[y_choice],
                zorder=8,
                mode="markers",
                name=region,
                customdata=sub.to_dict("records"),
                marker=dict(
                    color=REGION_COLORS[region],
                    size=10,
                    line=dict(width=1, color="white"),
                ),
                text="<b>" + sub["country"] + ":</b> " + sub["event"],
                hovertemplate=(
                    "%{text}<br>"
                    + y_choice
                    + ": %{y:,.0f}<br>"
                    + x_choice
                    + ": %{x:,.0f}<br>"
                    + "<extra></extra>"
                ),
            )
        )

    layout = go.Layout(
        xaxis=dict(title=x_choice, type="log"),
        yaxis=dict(title=y_choice, type="log"),
        legend=dict(title="Region"),
        margin=dict(l=20, r=20, t=20, b=20),
    )

    return traces, layout

