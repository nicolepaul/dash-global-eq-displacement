import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html
from dash.dash_table import DataTable
import dash_bootstrap_components as dbc

import statsmodels.formula.api as smf

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, root_mean_squared_error
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
            dbc.ListGroupItem(
                f"Root mean square error = {root_mean_squared_error(y, y_pred):.3f}"
            ),
            dbc.ListGroupItem(f"R² = {rsquared(y, y_pred):.1%}"),
            dbc.ListGroupItem(f"Consensus score = {custom_score(y, y_pred):.1%}"),
        ]
    except Exception as e:
        eval_items = [dbc.ListGroupItem(f"Error computing metrics: {e}")]

    return dbc.Row([dbc.ListGroup(eval_items, flush=True)])


def compute_interactions(model, X, features):
    explainer = shap.TreeExplainer(model)
    interaction_values = explainer.shap_interaction_values(X[features])
    mean_abs_interactions = np.abs(interaction_values).mean(axis=0)

    interaction_strength = pd.DataFrame(
        mean_abs_interactions, index=features, columns=features
    )
    np.fill_diagonal(interaction_strength.values, np.nan)
    return interaction_strength


def run_rfe(drivers, sub, metric, predictors, production=True):

    # Arrange response and explanatory variables
    X, y = sub[predictors], np.log(sub[metric].replace(0, FILL_ZERO))

    # Tune hyperparameters
    params = None
    mape_scorer = make_scorer(custom_score, greater_is_better=False)
    if production:  # hard-coded to save computation time
        if metric in PARAM_PROD:
            params = PARAM_PROD[metric]

        else:
            return (
                f"Production version hyperparameters not configured for {metric}",
                EMPTY_FIG,
                [],
            )

    else:
        grid = GridSearchCV(
            XGBRegressor(random_state=MODEL_SEED),
            param_grid=PARAM_GRID,
            cv=CV,
            scoring=mape_scorer,
            n_jobs=1,
        )
        grid.fit(X, y)
        params = grid.best_params_
    parm = summarize_model(params)

    # Use CV with RFE to select features; use repeats if not in production mode
    model = XGBRegressor(random_state=MODEL_SEED, **params)
    cv = KFold(n_splits=CV, shuffle=True, random_state=CV_SEED)
    if not production:
        cv = RepeatedKFold(n_splits=CV, n_repeats=S, random_state=CV_SEED)
    rfecv = RFECV(model, cv=cv, scoring=mape_scorer, step=1, n_jobs=1)
    rfecv.fit(X, y)
    selected = list(X.columns[rfecv.support_])

    # Arrange text summary
    summ = dbc.Row(
        [
            html.B(f"Selected {len(selected)} feature(s):"),
            dbc.ListGroup(
                [dbc.ListGroupItem(sel) for sel in selected],
                flush=True,
            ),
        ]
    )

    # Get out-of-fold predictions
    cv = KFold(n_splits=CV, shuffle=True, random_state=CV_SEED)
    oof_preds = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in cv.split(X):
        fold_model = XGBRegressor(**params)
        fold_model.fit(X.iloc[train_idx][selected], y.iloc[train_idx])
        oof_preds[test_idx] = fold_model.predict(X.iloc[test_idx][selected])
    eval_oof = summarize_evaluation(y, oof_preds)

    # Refit final model
    final_model = XGBRegressor(**params)
    final_model.fit(X[selected], y)

    # Evaluate model
    y_pred = final_model.predict(X[selected])
    eval_train = summarize_evaluation(y, y_pred)
    fig_eval = plot_model_eval(np.exp(y), np.exp(y_pred))
    summary = [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.ListGroup(
                            [
                                dbc.ListGroupItem(f"Number of events: {len(sub)}"),
                                dbc.ListGroupItem(
                                    f"Number of cross-validation folds: {CV}"
                                ),
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

    # Feature importance plot
    importances = final_model.feature_importances_
    fig_feature = plot_feature_importance(importances, selected)

    # PDPs for each selected features
    figs_pdp = []
    if not selected:
        return (
            [html.P("No features were selected. Try different predictors.")],
            EMPTY_FIG,
            [],
        )
    for feat in selected:
        figs_pdp.append(plot_pdp(final_model, X[selected], feat, drivers))

    # Compute interaction strengths
    interaction_df = compute_interactions(final_model, X[selected], selected)
    fig_interaction = plot_interactions(interaction_df)

    return summary, fig_feature, figs_pdp, fig_interaction


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


def bootstrap_uncertainty(X, y):

    coef_list = []
    int_list = []
    resid_list = []
    n = len(X)

    success = 0

    for _ in range(N_BOOTSTRAP):

        # Fit model
        idx = np.random.choice(n, size=n, replace=True)
        model = LinearRegression()
        model.fit(X.iloc[idx], y.iloc[idx])

        # Make predictions
        y_true, y_pred = y.iloc[idx], model.predict(X.iloc[idx])

        # Evaluate model
        r2 = rsquared(y_true, y_pred)
        mape = np.mean(percent_absolute_error(y_true, y_pred))
        mdape = np.median(percent_absolute_error(y_true, y_pred))

        # Only store results that pass certain criteria
        if (r2 > FILTER_R2) & (mape < FILTER_MAPE) & (mdape < FILTER_MDAPE):
            coef_list.append(model.coef_)
            int_list.append(model.intercept_)
            resid_list.append(y_true - y_pred)
            success += 1

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
                "b_ratio": success / N_BOOTSTRAP,
            }
        ),
        resid_arr,
    )


def fit_linear(data, metric, predictors, production=True):

    # Arrange response variable
    y = np.log(data[metric].replace(0, FILL_ZERO))

    # Initialize key variables
    subsets, best_subset, model_eval, model_uncertainty = None, None, None, None
    eval_test, eval_fig = None, None

    if production:

        # Bootstrapping best model predictor combination
        best_subset = LINEAR_PROD[metric]
        data = parse_transformations(data, best_subset)
        X = data[best_subset]
        X_best = data[best_subset]
        results = repeat_linear_models(X, y)
        eval_test = summarize_evaluation(
            np.concatenate(results["test_true"]), np.concatenate(results["test_pred"])
        )
        eval_fig = plot_model_eval_uncertainty(
            np.exp(np.concatenate(results["train_true"] + results["test_true"])),
            np.exp(np.concatenate(results["train_pred"] + results["test_pred"])),
            np.concatenate(results["train_idx"] + results["test_idx"]),
        )
        model_uncertainty, resid = bootstrap_uncertainty(X_best, y)

        # Load presaved model evaluation data
        model_eval = pd.read_csv(os.path.join("assets", f"linear_{metric}.csv"))
        subsets = model_eval["Permutation"].tolist()
        model_eval.set_index("Permutation", inplace=True)

    else:
        predictors = LINEAR_TERMS[metric]

        # Arrange different subsets of predictors
        data = parse_transformations(data, predictors)
        subsets = generate_predictor_subsets(predictors)
        subsets_tuple = [tuple(s) for s in subsets]

        # Initialize stores
        model_eval = pd.DataFrame(
            index=subsets_tuple,
            columns=["eval_r2", "eval_mape", "eval_mdape", "eval_consensus"],
        )

        # Iterate through each subset of predictors
        for subset in subsets_tuple:

            # Arrange explanatory variables
            X = data[list(subset)]

            # Repeat model fitting with different splits
            result = repeat_linear_models(X, y)
            eval_test = summarize_evaluation(
                np.concatenate(results["test_true"]),
                np.concatenate(results["test_pred"]),
            )
            eval_fig = plot_model_eval_uncertainty(
                np.exp(np.concatenate(results["train_true"] + results["test_true"])),
                np.exp(np.concatenate(results["train_pred"] + results["test_pred"])),
                np.concatenate(results["train_idx"] + results["test_idx"]),
            )

            # Store mean results
            for r in ["r2", "mape", "mdape", "consensus"]:
                model_eval.at[subset, f"eval_{r}"] = np.mean(results[f"eval_{r}"])

        # Manual ranking - select best model
        model_eval = model_eval.sort_values(by="eval_consensus", ascending=True)
        best_subset = model_eval.index[0]
        X_best = X[list(best_subset)]

        # Estimate uncertainty for best model
        model_uncertainty, resid = bootstrap_uncertainty(X_best, y)

    # Assemble final output
    summary = {
        "subset": list(best_subset),
        "eval": model_eval,
        "coef": model_uncertainty,
    }

    # Get residuals analysis - oof
    resid_oof = dbc.ListGroup(
        [
            dbc.ListGroupItem(f"Mean =  {resid.mean():.3f}"),
            dbc.ListGroupItem(f"Median =  {np.median(resid):.3f}"),
            dbc.ListGroupItem(f"Variance =  {np.var(resid):.3f}"),
            dbc.ListGroupItem(f"Standard deviation = {resid.std():.3f}"),
        ]
    )

    # Refit best model using median coefficients
    sel_coef = model_uncertainty["coef_mean"].values
    dummy_X = np.zeros((2, len(sel_coef) - 1))
    dummy_y = np.zeros(2)
    sel_model = LinearRegression().fit(dummy_X, dummy_y)  # dummy model to overwrite
    sel_model.coef_ = sel_coef[:-1]
    sel_model.intercept_ = sel_coef[-1]
    sel_model.feature_names_in_ = np.array(best_subset)
    predictors = model_uncertainty["Predictor"].values[:-1]
    sel_y_pred = sel_model.predict(X[predictors])
    eval_train = summarize_evaluation(y, sel_y_pred)

    # Get residuals analysis - median
    resid_meds = y - sel_y_pred
    resid_med = dbc.ListGroup(
        [
            dbc.ListGroupItem(f"Mean =  {resid_meds.mean():.3f}"),
            dbc.ListGroupItem(f"Median =  {np.median(resid_meds):.3f}"),
            dbc.ListGroupItem(f"Variance =  {np.var(resid_meds):.3f}"),
            dbc.ListGroupItem(f"Standard deviation = {resid_meds.std():.3f}"),
        ]
    )

    # Best model summary
    best_disp = model_uncertainty.copy()
    for col in best_disp.columns:
        if col != "Predictor":
            best_disp[col] = best_disp[col].apply(lambda x: f"{x:.3f}")
    eval_table = DataTable(
        data=best_disp.to_dict("records"),
        columns=[
            {"name": col.replace("coef_", ""), "id": col} for col in best_disp.columns
        ],
        style_cell={"fontSize": "0.75em"},
    )
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
                f"Out of all model permutations, the selected predictors are: {list(best_subset)}"
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.B("Model performance:"),
                            dbc.Row(
                                [
                                    dbc.Col([html.I("Bootstrap (out-of-fold)"), eval_test]),
                                    dbc.Col(
                                        [html.I("Mean coefficients"), eval_train]
                                    ),
                                ]
                            ),
                            html.P(),
                            html.B("Model coefficients"),
                            eval_table,
                            html.P(),
                            html.B("Residual analysis"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [html.I("Bootstrap (out-of-fold)"), resid_oof]
                                    ),
                                    dbc.Col([html.I("Mean coefficients"), resid_med]),
                                ]
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                dcc.Graph(figure=eval_fig),
                                html.I(
                                    "Note: Error bars in this plot represent the standard deviation from bootstrapping, and thus the uncertainty around the model's central estimates."
                                ),
                            ]
                        )
                    ),
                ]
            ),
        ]
    )

    # Model evaluation summary
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
