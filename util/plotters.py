import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import partial_dependence

from _config import *


def plot_scatter(df, y_choice, x_choice, z_choice):

    df = df.sort_values(by=z_choice)

    zs = df[z_choice].unique()
    REGION_COLORS = {zs[i]: CATEGORICAL_COLORS[i] for i in range(len(zs))}

    df["log_x"] = np.log(df[x_choice].replace(0, FILL_ZERO))
    df["log_y"] = np.log(df[y_choice].replace(0, FILL_ZERO))

    traces = []
    for z in zs:
        sub = df[df[z_choice] == z]
        traces.append(
            go.Scatter(
                x=sub[x_choice],
                y=sub[y_choice],
                zorder=8,
                mode="markers",
                name=z,
                customdata=sub.to_dict("records"),
                marker=dict(
                    color=REGION_COLORS[z],
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
        legend=dict(title=z_choice.capitalize()),
        margin=dict(l=20, r=20, t=20, b=20),
    )

    return traces, layout


def plot_model_eval(y, y_pred):

    fig_scatter = px.scatter(
        x=y,
        y=y_pred,
        labels={"x": "Observed", "y": "Predicted"},
    )

    fig_scatter.update_traces(
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
        height=400,
        width=400,
        yaxis={"type": "log"},
        xaxis={"type": "log"},
        margin=dict(l=60, r=40, t=40, b=40),
    )

    return fig_scatter


def plot_model_eval_uncertainty(y_true, y_pred, idx):

    all_data = pd.DataFrame({
        'idx': idx,
        'y_true': y_true,
        'y_pred': y_pred,
    })

    plot_data = all_data.groupby("idx").agg(
            y_true = ("y_true", "first"),
            y_pred_mean = ("y_pred", "mean"),
            y_pred_std = ("y_pred", "std"),
            y_pred_p10 = ("y_pred", lambda x: np.percentile(x, 10)),
            y_pred_p90 = ("y_pred", lambda x: np.percentile(x, 90)),
        ).reset_index()

    # fig_unc = px.violin(
    #         all_data,
    #         x="y_true",
    #         y="y_pred",
    #         box=True,
    #         labels={"y": "Observed", "y_pred": "Predicted"},
    #     )

    fig_unc = px.scatter(
            plot_data,
            x="y_true",
            y="y_pred_mean",
            error_y=plot_data["y_pred_std"],
            labels={"y_true": "Observed", "y_pred_mean": "Predicted"},
            custom_data=["y_pred_std", "y_pred_p10", "y_pred_p90"],
        )
    
    fig_unc.update_traces(
        marker={"size": 10, "line": dict(width=1, color="white")},
        hovertemplate="Observed: %{x:,.0f}<br>Predicted: %{y:,.0f}±%{customdata[0]:,.0f}",
    )
    
    fig_unc.add_shape(
        type="line",
        x0=min(y_true),
        x1=max(y_true),
        y0=min(y_true),
        y1=max(y_true),
        line=dict(color="silver", dash="dash"),
    )

    fig_unc.update_layout(
        height=400,
        width=400,
        yaxis={"type": "log"},
        xaxis={"type": "log"},
        margin=dict(l=60, r=40, t=40, b=40),
    )
    
    return fig_unc


def plot_corr_matx(sub, metric, drivers, method="pearson"):

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

    return dcc.Graph(figure=fig, style={"height": "750px"}), corr_df


def plot_hierarchical_cluster(sub, corr_method="pearson", link_method="ward"):

    corr = sub.corr(method=corr_method)
    dist = 1 - np.abs(corr)

    dist_condensed = squareform(dist.values, checks=False)
    Z = linkage(dist_condensed, method=link_method)

    fig = ff.create_dendrogram(
        dist.values,
        orientation="left",
        labels=dist.columns.tolist(),
        linkagefun=lambda _: Z,
        colorscale=CATEGORICAL_COLORS,
    )

    fig.update_layout(
        margin=dict(l=120, r=20, t=50, b=20),
        height=600,
        xaxis=dict(showticklabels=False, ticks=""),
        yaxis=dict(showticklabels=True, ticks=""),
    )

    return dcc.Graph(figure=fig, style={"height": "600px"})


def plot_mutual_information(sub, metric):

    sub = sub.dropna()
    X = sub.drop(columns=metric)
    y = sub[metric]

    mi_scores = mutual_info_regression(X, y)
    mi_df = pd.DataFrame(
        {"Explanatory variable": X.columns, "Mutual information": mi_scores}
    )
    mi_df = mi_df.sort_values("Mutual information")

    fig = px.bar(
        mi_df,
        x="Mutual information",
        y="Explanatory variable",
        orientation="h",
    )

    fig.update_traces(hovertemplate="<b>%{y}:</b><br>%{x:.0%}<extra></extra>")

    fig.update_layout(yaxis_title=None)

    return dcc.Graph(figure=fig, style={"height": "600px"}), mi_df


def plot_feature_importance(importances, selected):
    fig_imp = go.Figure(
        [go.Bar(x=importances, y=selected, orientation="h",
                hovertemplate="<b>%{y}: </b>%{x:.1%}<extra></extra>")]
    )
    fig_imp.update_layout(yaxis=dict(autorange="reversed"))
    return dcc.Graph(figure=fig_imp, style={"height": "400px"})


def plot_pdp(final_model, X_selected, feat, drivers):
    try:
        pred = partial_dependence(final_model, X_selected, feat, kind="both")
        ice_curves = np.squeeze(pred["individual"])
        grid = pred["grid_values"][0] if isinstance(pred["grid_values"], (list, tuple)) else pred["grid_values"]
    except Exception:
        return dcc.Graph(figure=EMPTY_FIG, style={"height": "300px"})

    fig = go.Figure()
    if ice_curves is not None and ice_curves.ndim == 2:
        for sample_curve in ice_curves:
            fig.add_trace(
                go.Scatter(x=grid, y=sample_curve, mode="lines", line=dict(width=1, color="silver"), hoverinfo="skip")
            )

    avg = ice_curves.mean(axis=0) if ice_curves is not None else np.zeros_like(grid)
    fig.add_trace(
        go.Scatter(x=grid, y=avg, mode="lines", line=dict(width=3, color="#212121"), name="Average")
    )

    title = drivers.loc[drivers.variable == feat, "name"].values
    title = title[0] if len(title) else feat

    fig.update_layout(title=title, showlegend=False, margin=dict(l=40, r=20, t=30, b=30), height=300)
    return dcc.Graph(figure=fig, style={"height": "300px"})


def plot_interactions(interaction_df):

    fig = px.imshow(
        interaction_df,
        color_continuous_scale="RdBu",
        aspect="auto",
        zmin=-1, # red should be out of scale
        zmax=1,
        text_auto=".0%",
    )
    fig.update_traces(hovertemplate="%{x}, %{y}:<br><b>%{z:.0%}</b><extra></extra>")
    fig.update_layout(margin=dict(l=60, r=40, t=40, b=40), coloraxis={"showscale": False})
    return dcc.Graph(figure=fig, style={"height": "600px"})