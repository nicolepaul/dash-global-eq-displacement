import numpy as np
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, dcc, ALL, ctx, no_update

from util.plotters import plot_corr_matx, plot_hierarchical_cluster, plot_mutual_information
from util.analysis import run_rfe
from _config import *


def drivers_table(drivers):

    header = [
        html.Thead(
            html.Tr([html.Th("Driver"), html.Th("Variable name"), html.Th("Source")])
        )
    ]

    rows = []
    for _, driver in drivers.iterrows():
        rows.append(
            html.Tr(
                [
                    html.Td(driver["name"]),
                    html.Td(driver["variable"]),
                    html.Td(dcc.Markdown(driver["source"])),
                ]
            )
        )
    body = [html.Tbody(rows)]

    return dbc.Accordion(
        [
            dbc.AccordionItem(
                dbc.Table(header + body),
                title="Data definitions and sources",
            ),
        ],
        start_collapsed=True,
    )


def summary_default(drivers, fig_corr, fig_cluster, fig_mi):

    narrative = html.Div(
        [
            drivers_table(drivers),
            html.P(),
            html.H4("Correlation analysis"),
            html.P(NARRATIVE_CORR),
            fig_corr,
            html.Hr(),
            html.H4('Hierarchical clustering'),
            html.P(NARRATIVE_HIER),
            fig_cluster,
            html.Hr(),
            html.H4('Mutual information'),
            html.P(NARRATIVE_MI),
            fig_mi,
        ]
    )

    return narrative


def summary_rfe(summary, plot_feature, plot_pdp):

    narrative = html.Div(
        [
            html.H3("Feature selection"),
            html.P(NARRATIVE_FS),
            html.H4("Recursive feature elimination"),
            html.P(NARRATIVE_RFE),
            html.Div(summary),
            html.Hr(),
            html.H4("Feature importance"),
            html.P(NARRATIVE_FI),
            plot_feature,
            html.Hr(),
            html.H4("Partial dependence plots"),
            html.P(NARRATIVE_PDP),
            html.Div(plot_pdp),
        ]
    )

    return narrative


def register_callbacks_drivers(app, df, drivers, production=True):

    @app.callback(
        Output("analysis-btn", "disabled", allow_duplicate=True),
        Input("analysis-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def disable_btn(n):
        if n:
            return True
        return no_update
    

    @app.callback(
        Output("drivers-results-container", "children"),
        Output("analysis-btn", "disabled"),
        Input("analysis-btn", "n_clicks"),
        Input("explore-btn", "n_clicks"),
        Input("tab-content", "children"),
        State("rfe-metric-dropdown", "value"),
        State({"type": "driver-checkbox", "index": ALL}, "value"),
        State({"type": "driver-checkbox", "index": ALL}, "id"),
        prevent_initial_call=False,
    )
    def run_analysis(clicks_1, clicks_2, tab_children, metric, values, ids):

        triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else None

        # Get predictors
        predictors = [id_["index"] for val, id_ in zip(values, ids) if val]

        # On load or 'Explore dirvers' click: correlation
        if (not predictors or not triggered.startswith("analysis-btn")) | triggered.startswith("explore-btn"):
            predictors = [id_["index"] for val, id_ in zip(values, ids) if val]
            if not predictors:
                return (
                    html.P("No explanatory variables selected for correlation matrix."),
                    False,
                )
            corr_vars = [
                c
                for c in predictors
                if c in df.columns and np.issubdtype(df[c].dtype, np.number)
            ]
            if not corr_vars:
                return (
                    html.P("No numeric variables available for correlation matrix."),
                    False,
                )
            fig_corr, corr = plot_corr_matx(df[[metric] + corr_vars].copy(), metric, drivers)
            fig_cluster = plot_hierarchical_cluster(df[corr_vars].copy())
            fig_mi, mi = plot_mutual_information(df[[metric] + corr_vars].copy(), metric)

            # Print to log redundant features
            upper = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            redundant = [column for column in upper.columns if any(upper[column] > CORR_THRESH)]
            corr_keep = [v for v in corr.columns if v not in redundant]
            mi_cutoff = mi["Mutual information"].quantile(MI_QUANT)
            mi_keep = mi.loc[mi["Mutual information"] >= mi_cutoff, "Explanatory variable"].tolist()
            keep = list(set(mi_keep) & set(corr_keep))
            print(f"Selected {len(keep)} features: {keep}")

            return summary_default(drivers, fig_corr, fig_cluster, fig_mi), False

        # On 'Run analysis' click: RFE
        if not metric or not predictors:
            return (
                html.P(
                    "Please select a metric and predictors before running analysis."
                ),
                False,
            )
        sub = df[[metric] + predictors].dropna()
        n_sample = len(sub)
        if n_sample < MIN_EVENT:
            return (
                html.Div(
                    f"Insufficient events ({n_sample}, need minimum {MIN_EVENT})."
                ),
                False,
            )

        try:
            summary, plot_feature, plot_pdp = run_rfe(drivers, sub, metric, predictors, production=True)
        except Exception as e:
            return html.Div([html.P("Error running analysis"), html.Pre(str(e))]), False

        results = summary_rfe(summary, plot_feature, plot_pdp)

        return results, False
