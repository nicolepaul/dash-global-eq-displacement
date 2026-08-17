import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html
from dash.dash_table import DataTable
import dash_bootstrap_components as dbc

import statsmodels.formula.api as smf

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, GridSearchCV, train_test_split
from sklearn.metrics import make_scorer
from sklearn.feature_selection import RFECV

from util.transform import (
    parse_transformations,
    generate_predictor_subsets,
)

import shap


from _config import *
from util.metrics import (
    rsquared,
    percent_absolute_error,
    custom_score,
)
from util.plotters import (
    plot_model_eval,
    plot_feature_importance,
    plot_pdp,
    plot_interactions,
    plot_model_eval_uncertainty,
    plot_residuals,
)


def run_regression(df, x_col, y_col, method="ols", add_trace=True):

    N_PLOT = 200

    df[x_col] = np.maximum(df[x_col], FILL_ZERO)
    df[y_col] = np.maximum(df[y_col], FILL_ZERO)
    df["log_x"] = np.log(df[x_col])
    df["log_y"] = np.log(df[y_col])

    eqn = None
    lm_fit = None
    color, style = "black", "dash"
    if method == "ols":
        lm = smf.ols(formula="log_y ~ log_x - 1", data=df)
        lm_fit = lm.fit()
        eqn = html.Div(
            [
                dcc.Markdown("$\\log y=\\beta \\cdot \\log x$", mathjax=True),
                dcc.Markdown("Or, $y = x^\\beta$", mathjax=True),
            ]
        )
    elif method == "ols_int":
        lm = smf.ols(formula="log_y ~ log_x", data=df)
        lm_fit = lm.fit()
        eqn = html.Div(
            [
                dcc.Markdown("$\\log y=\\beta \\cdot \\log x + \\alpha$", mathjax=True),
                dcc.Markdown("Or, $y = e^{\\alpha}x^{\\beta}$", mathjax=True),
            ]
        )
    elif method == "rlm":
        lm = smf.rlm(formula="log_y ~ log_x - 1", data=df)
        lm_fit = lm.fit()
        eqn = html.Div(
            [
                dcc.Markdown("$\\log y=\\beta \\cdot \\log x$", mathjax=True),
                dcc.Markdown("Or, $y = x^\\beta$", mathjax=True),
            ]
        )
        color, style = "darkred", "dot"
    elif method == "rlm_int":
        lm = smf.rlm(formula="log_y ~ log_x", data=df)
        lm_fit = lm.fit()
        eqn = html.Div(
            [
                dcc.Markdown("$\\log y=\\beta \\cdot \\log x + \\alpha$", mathjax=True),
                dcc.Markdown("Or, $y = e^{\\alpha}x^{\\beta}$", mathjax=True),
            ]
        )
        color, style = "darkred", "dot"
    else:
        raise NotImplementedError(f"Method '{method}' not supported yet.")

    # Get key values
    alpha = lm_fit.params["Intercept"] if hasattr(lm_fit.params, "Intercept") else None
    beta = lm_fit.params["log_x"] if hasattr(lm_fit.params, "log_x") else None
    df["pred_y"] = lm_fit.predict(df["log_x"])

    # Write narrative
    narrative = []
    if beta is not None:

        regression_results = dbc.ListGroup(
            [
                html.I("Model support"),
                dbc.ListGroupItem(f"n = {df[y_col].count():,.0f}"),
                html.I("Model coefficient(s)"),
                dbc.ListGroupItem(f"α = {alpha:.3f}" if alpha is not None else ""),
                dbc.ListGroupItem(f"β = {beta:.3f}" if beta is not None else ""),
                html.I("Model evaluation (training set)"),
                dbc.ListGroupItem(f"R² = {rsquared(df['log_y'], df['pred_y']):.1%}"),
                dbc.ListGroupItem(
                    f"Median abs. percent error = {np.median(percent_absolute_error(df['log_y'], df['pred_y'])):.1%}"
                ),
                dbc.ListGroupItem(
                    f"Mean abs. percent error = {np.mean(percent_absolute_error(df['log_y'], df['pred_y'])):.1%}"
                ),
                dbc.ListGroupItem(
                    f"Consensus score = {custom_score(df['log_y'], df['pred_y']):.1%}"
                ),
            ],
            flush=True,
        )
        narrative = [
            html.P(),
            eqn,
            html.B("Regression Results"),
            regression_results,
        ]

    # Define trace
    trace = None
    if add_trace:
        log_x = np.linspace(np.log(FILL_ZERO), df["log_x"].max(), N_PLOT)
        y_hat = lm_fit.predict(exog=dict(log_x=log_x))

        trace = go.Scatter(
            x=np.exp(log_x),
            y=np.exp(y_hat),
            mode="lines",
            name=f"{method.upper()} (β={beta:.3f})",
            line=dict(color=color, dash=style),
            hoverinfo="skip",
            zorder=5,
        )

    return trace, narrative


def summarize_model(params):
    if params is None:
        return html.Div()
    items = []
    items.append(
        dbc.ListGroupItem(f"Number of estimators = {params.get('n_estimators','')}")
    )
    items.append(dbc.ListGroupItem(f"Max depth = {params.get('max_depth','')}"))
    items.append(dbc.ListGroupItem(f"Learning rate = {params.get('learning_rate','')}"))
    items.append(dbc.ListGroupItem(f"Gamma = {params.get('gamma','')}"))
    items.append(dbc.ListGroupItem(f"Alpha = {params.get('reg_alpha','')}"))
    items.append(dbc.ListGroupItem(f"Lambda = {params.get('reg_lambda','')}"))
    items.append(
        dbc.ListGroupItem(f"Min child weight = {params.get('min_child_weight','')}")
    )
    return dbc.Row([html.B("Model hyperparameters:"), dbc.ListGroup(items, flush=True)])


def summarize_evaluation(y, y_pred):
    try:
        eval_items = [
            dbc.ListGroupItem(
                f"Median abs. percent error = {np.median(percent_absolute_error(y, y_pred)):.1%}"
            ),
            dbc.ListGroupItem(
                f"Mean abs. percent error = {np.mean(percent_absolute_error(y, y_pred)):.1%}"
            ),
            dbc.ListGroupItem(f"R² = {rsquared(y, y_pred):.1%}"),
            dbc.ListGroupItem(f"Consensus score = {custom_score(y, y_pred):.1%}"),
        ]
    except Exception as e:
        eval_items = [dbc.ListGroupItem(f"Error computing metrics: {e}")]

    return dbc.Row([dbc.ListGroup(eval_items, flush=True)])


def summarize_residuals(resid):
    return dbc.ListGroup(
        [
            dbc.ListGroupItem(f"Mean = {np.mean(resid):.3f}"),
            dbc.ListGroupItem(f"Median = {np.median(resid):.3f}"),
            dbc.ListGroupItem(f"Variance = {np.var(resid):.3f}"),
            dbc.ListGroupItem(f"Standard deviation = {np.std(resid):.3f}"),
        ]
    )


def compute_interactions(model, X, features):
    explainer = shap.TreeExplainer(model)
    interaction_values = explainer.shap_interaction_values(X[features])
    mean_abs_interactions = np.abs(interaction_values).mean(axis=0)

    interaction_strength = pd.DataFrame(
        mean_abs_interactions, index=features, columns=features
    )
    np.fill_diagonal(interaction_strength.values, np.nan)
    return interaction_strength


def run_tree_rfe(drivers, sub, metric, predictors, production=True):
    fit = specify_tree(sub, metric, predictors, production=production)
    if fit is None:
        return (
            f"Production version hyperparameters or terms not configured for {metric}",
            EMPTY_FIG,
            [],
            [],
        )
    if not fit["selected"]:
        return (
            [html.P("No features were selected. Try different predictors.")],
            EMPTY_FIG,
            [],
            [],
        )
    evaluation = evaluate_tree(fit)
    return render_tree_summary(drivers, fit, evaluation)


def specify_tree(sub, metric, predictors, production=True):

    X, y = sub[predictors], np.log(sub[metric].replace(0, FILL_ZERO))
    cv = RepeatedKFold(n_splits=CV, n_repeats=S, random_state=CV_SEED)
    mape_scorer = make_scorer(custom_score, greater_is_better=False)

    if production:
        if metric not in PARAM_PROD:
            return None
        params = PARAM_PROD[metric]
        selected = TREE_PROD[metric]
    else:
        print("running in non-production mode (will take some time)")
        grid = GridSearchCV(
            XGBRegressor(random_state=MODEL_SEED),
            param_grid=PARAM_GRID,
            cv=cv,
            scoring=mape_scorer,
            n_jobs=1,
        )
        grid.fit(X, y)
        params = grid.best_params_

        model = XGBRegressor(random_state=MODEL_SEED, **params)
        rfecv = RFECV(model, cv=cv, scoring=mape_scorer, step=1, n_jobs=1)
        rfecv.fit(X, y)
        selected = list(X.columns[rfecv.support_])

    final_model = XGBRegressor(random_state=MODEL_SEED, **params)
    final_model.fit(X[selected], y)

    return {
        "X": X,
        "y": y,
        "cv": cv,
        "params": params,
        "selected": selected,
        "final_model": final_model,
    }


def evaluate_tree(fit):

    X, y, cv = fit["X"], fit["y"], fit["cv"]
    selected, params, final_model = fit["selected"], fit["params"], fit["final_model"]

    return {
        "oof_preds": compute_oof_predictions(X, y, selected, params, cv),
        "y_pred": final_model.predict(X[selected]),
        "importances": final_model.feature_importances_,
        "interaction_df": compute_interactions(final_model, X[selected], selected),
    }


def render_tree_summary(drivers, fit, evaluation):

    X, y, selected, params, final_model = (
        fit["X"],
        fit["y"],
        fit["selected"],
        fit["params"],
        fit["final_model"],
    )
    oof_preds, y_pred = evaluation["oof_preds"], evaluation["y_pred"]

    parm = summarize_model(params)
    summ = dbc.Row(
        [
            html.B(f"Selected {len(selected)} feature(s):"),
            dbc.ListGroup([dbc.ListGroupItem(sel) for sel in selected], flush=True),
        ]
    )
    eval_oof = summarize_evaluation(y, oof_preds)
    eval_train = summarize_evaluation(y, y_pred)
    fig_eval = plot_model_eval(np.exp(y), np.exp(y_pred))

    summary = [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.ListGroup(
                            [
                                dbc.ListGroupItem(f"Number of events: {len(X)}"),
                                dbc.ListGroupItem(
                                    f"Number of cross-validation folds: {CV}"
                                ),
                                dbc.ListGroupItem(f"Number of sample repeats: {S}"),
                                html.P(),
                            ]
                        ),
                        dbc.Row([dbc.Col(summ), dbc.Col(parm)]),
                        html.P(),
                        html.B("Model performance:"),
                        dbc.Row(
                            [
                                dbc.Col([html.I("CV (out-of-fold)"), eval_oof]),
                                dbc.Col([html.I("Refit (training)"), eval_train]),
                            ]
                        ),
                    ],
                    md=6,
                ),
                dbc.Col(dcc.Graph(figure=fig_eval)),
            ]
        )
    ]

    fig_feature = plot_feature_importance(evaluation["importances"], selected)
    figs_pdp = [plot_pdp(final_model, X[selected], feat, drivers) for feat in selected]
    fig_interaction = plot_interactions(evaluation["interaction_df"])

    return summary, fig_feature, figs_pdp, fig_interaction


def compute_oof_predictions(X, y, selected, params, cv, seed=MODEL_SEED):
    oof_sum, oof_count = np.zeros(len(y)), np.zeros(len(y))
    for train_idx, test_idx in cv.split(X):
        fold_model = XGBRegressor(random_state=seed, **params)
        fold_model.fit(X.iloc[train_idx][selected], y.iloc[train_idx])
        preds = fold_model.predict(X.iloc[test_idx][selected])
        oof_sum[test_idx] += preds
        oof_count[test_idx] += 1
    return oof_sum / oof_count


def repeat_linear_models(X, y, seed=42):

    n = len(X)

    r2, mape, mdape, custom = [], [], [], []
    train_pred, train_true, test_pred, test_true = [], [], [], []
    train_idxs, test_idxs = [], []

    for rep in range(LIN_REPEATS):
        train_idx, test_idx = train_test_split(
            np.arange(n), test_size=LIN_TEST, random_state=seed + rep
        )

        model = LinearRegression()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        y_pred = model.predict(X.iloc[test_idx])
        y_test = y.iloc[test_idx]

        r2.append(rsquared(y_test, y_pred))
        mape.append(np.mean(percent_absolute_error(y_test, y_pred)))
        mdape.append(np.median(percent_absolute_error(y_test, y_pred)))
        custom.append(custom_score(y_test, y_pred))
        train_pred.append(model.predict(X.iloc[train_idx]))
        train_true.append(y.iloc[train_idx])
        test_pred.append(y_pred)
        test_true.append(y_test)
        train_idxs.append(train_idx)
        test_idxs.append(test_idx)

    return {
        "eval_r2": r2,
        "eval_mape": mape,
        "eval_mdape": mdape,
        "eval_consensus": custom,
        "train_pred": train_pred,
        "train_true": train_true,
        "train_idx": train_idxs,
        "test_pred": test_pred,
        "test_true": test_true,
        "test_idx": test_idxs,
    }


def bootstrap_uncertainty(X, y, random_state=303):

    np.random.seed(random_state)

    coef_list = []
    int_list = []
    resid_list = []
    n = len(X)

    for _ in range(N_BOOTSTRAP):

        # Fit model
        idx = np.random.choice(n, size=n, replace=True)
        model = LinearRegression()
        model.fit(X.iloc[idx], y.iloc[idx])

        # Make predictions
        y_true, y_pred = y.iloc[idx], model.predict(X.iloc[idx])

        # Store results
        coef_list.append(model.coef_)
        int_list.append(model.intercept_)
        resid_list.append(y_true - y_pred)

    coef_arr = np.array(coef_list)
    int_arr = np.array(int_list)
    resid_arr = np.array(resid_list)

    return (
        pd.DataFrame.from_dict(
            {
                "Predictor": X.columns.tolist() + ["Intercept"],
                "coef_mean": np.concatenate([coef_arr.mean(axis=0), [int_arr.mean()]]),
                "coef_median": np.concatenate(
                    [np.median(coef_arr, axis=0), [np.median(int_arr)]]
                ),
                "coef_std": np.concatenate([coef_arr.std(axis=0), [int_arr.std()]]),
                "lower": np.concatenate(
                    [np.percentile(coef_arr, 10, axis=0), [np.percentile(int_arr, 10)]]
                ),
                "upper": np.concatenate(
                    [np.percentile(coef_arr, 90, axis=0), [np.percentile(int_arr, 90)]]
                ),
            }
        ),
        resid_arr,
    )

def fit_linear(drivers, data, metric, predictors, production=True):
    fit = specify_linear(data, metric, predictors, production=production)
    evaluation = evaluate_linear(fit)
    return render_linear_summary(drivers, fit, evaluation)


def specify_linear(data, metric, predictors, production=True):

    y = np.log(data[metric].replace(0, FILL_ZERO))
    predictors = LINEAR_PROD[metric] if production else LINEAR_TERMS[metric]
    data_transformed = parse_transformations(data, predictors)

    if production:
        best_subset = LINEAR_PROD[metric]
        model_eval = pd.read_csv(os.path.join("assets", f"linear_{metric}.csv"))
        subsets = model_eval["Permutation"].tolist()
        model_eval.set_index("Permutation", inplace=True)

    else:
        print("fitting in non-production mode; may take some minutes...")
        subsets = generate_predictor_subsets(predictors, max_terms=MAX_TERMS)
        subsets_tuple = [tuple(s) for s in subsets]

        model_eval = pd.DataFrame(
            index=subsets_tuple,
            columns=["eval_r2", "eval_mape", "eval_mdape", "eval_consensus"],
        )
        for subset in subsets_tuple:
            X = data_transformed[list(subset)].copy()
            subset_results = repeat_linear_models(X, y)
            pooled_true = np.concatenate(subset_results["test_true"])
            pooled_pred = np.concatenate(subset_results["test_pred"])
            model_eval.at[subset, "eval_r2"] = rsquared(pooled_true, pooled_pred)
            model_eval.at[subset, "eval_mape"] = np.mean(percent_absolute_error(pooled_true, pooled_pred))
            model_eval.at[subset, "eval_mdape"] = np.median(percent_absolute_error(pooled_true, pooled_pred))
            model_eval.at[subset, "eval_consensus"] = custom_score(pooled_true, pooled_pred)

        model_eval = model_eval.sort_values(by="eval_consensus", ascending=True)
        model_eval.to_csv(os.path.join("assets", f"linear_{metric}.csv"), index_label="Permutation") # not relevant for prod mode
        best_subset = model_eval.index[0]
        subsets = subsets_tuple

    X_best = data_transformed[list(best_subset)].copy()
    model_uncertainty, _ = bootstrap_uncertainty(X_best, y)

    return {
        "y": y,
        "data_transformed": data_transformed,
        "best_subset": list(best_subset),
        "model_eval": model_eval,
        "model_uncertainty": model_uncertainty,
        "X_best": X_best,
        "subsets": subsets,
    }


def evaluate_linear(fit):

    y, data_transformed, X_best = fit["y"], fit["data_transformed"], fit["X_best"]
    best_predictors = fit["model_uncertainty"]["Predictor"].values[:-1]

    results = repeat_linear_models(X_best, y)
    resid_hos = np.concatenate(results["test_true"]) - np.concatenate(
        results["test_pred"]
    )

    sel_model = LinearRegression().fit(X_best, y)
    sel_y_pred = sel_model.predict(data_transformed[best_predictors])
    sel_coef = np.append(sel_model.coef_, sel_model.intercept_)
    resid_ffs = y - sel_y_pred

    model_uncertainty = fit["model_uncertainty"].copy()
    model_uncertainty["full_fit"] = sel_coef

    return {
        "results": results,
        "resid_hos": resid_hos,
        "sel_y_pred": sel_y_pred,
        "resid_ffs": resid_ffs,
        "model_uncertainty": model_uncertainty,
        "best_predictors": best_predictors,
    }


def render_linear_summary(drivers, fit, evaluation):

    # Arrange information
    y, data_transformed, X_best = fit["y"], fit["data_transformed"], fit["X_best"]
    best_subset, model_eval, subsets = (
        fit["best_subset"],
        fit["model_eval"],
        fit["subsets"],
    )

    results, resid_hos = evaluation["results"], evaluation["resid_hos"]
    sel_y_pred, resid_ffs = evaluation["sel_y_pred"], evaluation["resid_ffs"]
    model_uncertainty, best_predictors = (
        evaluation["model_uncertainty"],
        evaluation["best_predictors"],
    )

    summary = {"subset": best_subset, "eval": model_eval, "coef": model_uncertainty}

    # Held out set
    eval_test = summarize_evaluation(
        np.concatenate(results["test_true"]), np.concatenate(results["test_pred"])
    )
    eval_fig_ho = plot_model_eval_uncertainty(
        np.exp(np.concatenate(results["test_true"])),
        np.exp(np.concatenate(results["test_pred"])),
        np.concatenate(results["test_idx"]),
    )
    resid_ho = summarize_residuals(resid_hos)

    # Proposed model (full fit)
    eval_train = summarize_evaluation(y, sel_y_pred)
    eval_fig_ff = plot_model_eval(np.exp(y), np.exp(sel_y_pred))
    resid_ff = summarize_residuals(resid_ffs)

    figs_resid = [
        plot_residuals(resid_ffs, X_best[feat], feat, drivers)
        for feat in best_predictors
    ]

    # Bootstrapping results with proposed model (full fit)
    best_disp = model_uncertainty.copy()
    for col in best_disp.columns:
        if col != "Predictor":
            best_disp[col] = best_disp[col].apply(lambda x: f"{x:.3f}")
    eval_table = DataTable(
        data=best_disp.to_dict("records"),
        columns=[
            {
                "name": "proposed" if col == "full_fit" else col.replace("coef_", ""),
                "id": col,
            }
            for col in best_disp.columns
        ],
        style_cell={"fontSize": "0.75em"},
        style_data_conditional=[
            {"if": {"column_id": "full_fit"}, "fontWeight": "bold"}
        ],
    )
    coef_range = dbc.ListGroup(
        [
            dbc.ListGroupItem(
                f"{data_transformed[coef].min():.2f} ≤ {coef} ≤ {data_transformed[coef].max():.2f}"
            )
            for coef in best_disp["Predictor"]
            if coef != "Intercept"
        ]
    )

    # Construct summary
    best = html.Div(
        [
            dbc.ListGroup(
                [
                    dbc.ListGroupItem(f"Test proportion: {LIN_TEST}"),
                    dbc.ListGroupItem(f"Number of repeats: {LIN_REPEATS}"),
                    dbc.ListGroupItem(f"Number of bootstrap samples: {N_BOOTSTRAP}"),
                    html.P(),
                ]
            ),
            html.P(
                f"Out of all model permutations, the selected predictors are: {best_subset}"
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.B("Model coefficients"),
                            eval_table,
                            html.P(),
                            html.B("Predictor ranges"),
                            coef_range,
                            html.P(),
                            html.B("Model evaluation"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.I("Held-out splits"),
                                            eval_test,
                                            dcc.Graph(figure=eval_fig_ho),
                                        ]
                                    ),
                                    dbc.Col(
                                        [
                                            html.I("Proposed coefficients (full fit)"),
                                            eval_train,
                                            dcc.Graph(figure=eval_fig_ff),
                                        ]
                                    ),
                                ]
                            ),
                            html.P(),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.B("Residual analysis"),
                                dbc.Row(
                                    [
                                        dbc.Col([html.I("Held-out splits"), resid_ho]),
                                        dbc.Col(
                                            [
                                                html.I(
                                                    "Proposed coefficients (full fit)"
                                                ),
                                                resid_ff,
                                            ]
                                        ),
                                    ]
                                ),
                                html.P(),
                                html.Div(figs_resid),
                            ]
                        )
                    ),
                ]
            ),
        ]
    )

    # Permutation summary
    n_permutations = len(subsets)
    eval_disp = model_eval.reset_index().rename(columns={"index": "Permutation"})
    eval_disp["Permutation"] = eval_disp["Permutation"].apply(
        lambda x: ", ".join(x) if isinstance(x, (tuple, list)) else str(x)
    )
    for col in eval_disp.columns:
        if col != "Permutation":
            eval_disp[col] = eval_disp[col].apply(lambda x: f"{x:.3f}")
    perm = html.Div(
        [
            html.P(
                f"A total of {n_permutations:,.0f} unique combinations of predictors were investigated"
            ),
            DataTable(
                data=eval_disp.to_dict("records"),
                columns=[
                    {"name": col.replace("eval_", ""), "id": col}
                    for col in eval_disp.columns
                    if col.startswith("eval_") or col == "Permutation"
                ],
                page_size=50,
                style_cell={"fontSize": "0.75em"},
            ),
        ]
    )

    return summary, best, perm
