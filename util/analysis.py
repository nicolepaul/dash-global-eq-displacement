import numpy as np
import statsmodels.formula.api as smf
import plotly.graph_objs as go

def run_regression(df, x_col, y_col, method="ols", add_trace=True):

    N = 200

    df["log_x"] = np.log10(df[x_col])
    df["log_y"] = np.log10(df[y_col])

    # NOTE: Fitting OLS without an intercept
    if method == "ols":
        lm = smf.ols(formula="log_y ~ log_x - 1", data=df)
        lm_fit = lm.fit()
    else:
        raise NotImplementedError(f"Method '{method}' not supported yet.")

    # Get key values
    alpha = lm_fit.params["log_x"]
    r2 = lm_fit.rsquared if hasattr(lm_fit, "rsquared") else None

    # Define trace
    trace = None
    if add_trace:
        # Generate prediction grid
        x_vals = np.logspace(df["log_x"].min(), df["log_x"].max(), N, base=10)
        log_x = np.log10(x_vals)
        y_hat = lm_fit.predict(exog=dict(log_x=log_x))

        trace = go.Scatter(
            x=x_vals,
            y=10 ** y_hat,
            mode="lines",
            name=f"{method.upper()} (α={alpha:.2f})",
            line=dict(color="black", dash="dash"),
        )

    return alpha, r2, trace
