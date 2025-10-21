import numpy as np
import plotly.graph_objs as go
import plotly.express as px

from _config import *


def arrange_scatter(df, y_choice, x_choice):

    regions = df["region"].unique()
    REGION_COLORS = {regions[i]: CATEGORICAL_COLORS[i] for i in range(len(regions))}

    df["log_x"] = np.log1p(df[x_choice])
    df["log_y"] = np.log1p(df[y_choice])

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


def arrange_corr_matx(sub, metric, drivers, method="pearson"):

    # Rename metric for pretty display
    sub.rename(columns={metric: metric.split("_").pop(0).upper()}, inplace=True)
    metric = metric.split("_").pop(0).upper()

    # Arrange correlation matrix
    corr_df = sub.corr(method=method)
    metric_row = corr_df.loc[[metric], corr_df.columns.drop(metric)]
    corr_df = corr_df.drop(index=metric).drop(columns=[metric])

    # Order variables by category in case they become jumbled
    ordered_vars = []
    for cat in drivers["category"].dropna().unique():
        cat_vars = drivers.loc[drivers["category"] == cat, "variable"]
        ordered_vars.extend([v for v in cat_vars if v in corr_df.columns])
    corr_df = corr_df.loc[ordered_vars, ordered_vars]
    metric_row = metric_row[ordered_vars]

    # Arrange main display
    display_corr = np.vstack([metric_row.values, corr_df.values])
    display_index = [metric] + list(corr_df.index)
    fig = px.imshow(
        display_corr,
        x=corr_df.columns,
        y=display_index,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        text_auto=".0%",
        aspect="auto",
    )

    n_cols = len(corr_df.columns)

    # Thick border for displacement metric
    fig.add_shape(
        type="rect",
        x0=-0.5,
        x1=n_cols - 0.5,
        y0=-0.5,
        y1=0.5,
        line=dict(color="black", width=3),
    )

    # Other layout adjustments
    fig.update_traces(hovertemplate="%{x}, %{y}:<br><b>%{z:.0%}</b><extra></extra>")
    fig.update_layout(
        coloraxis={"showscale": False},
        margin=dict(l=60, r=40, t=40, b=40),
    )

    return dcc.Graph(figure=fig, style={"height": "750px"})


def plot_model_eval(y, y_pred):

    fig_scatter = px.scatter(
        x=y,
        y=y_pred,
        labels={"x": "Observed", "y": "Predicted"},
    ).update_traces(
        marker={"size": 10, "line": dict(width=1, color="white")},
        hovertemplate="Observed: %{x:,.0f}<br>Predicted: %{y:,.0f}",
    )
    fig_scatter.add_shape(
        type="line",
        x0=min(y),
        x1=max(y),
        y0=min(y),
        y1=max(y),
        line=dict(color="silver", dash="dash"),
    )
    fig_scatter.update_layout(
        height=500,
        width=500,
        yaxis={"type": "log"},
        xaxis={"type": "log"},
        margin=dict(l=60, r=40, t=40, b=40),
    )

    return fig_scatter
