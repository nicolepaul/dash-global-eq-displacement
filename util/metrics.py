import numpy as np

def percent_error(y_true, y_pred):
    idx = (y_true > 1e-6)
    return (y_true[idx] - y_pred[idx]) / y_true[idx]

def percent_absolute_error(y_true, y_pred):
    idx = (y_true > 1e-6)
    return absolute_error(y_true[idx], y_pred[idx]) / np.abs(y_true[idx])

def absolute_error(y_true, y_pred):
    return np.abs((y_true - y_pred))

def rsquared(y_true, y_pred):
    rss = np.sum((y_true - y_pred) ** 2)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (rss / tss)

def custom_score(y_true, y_pred):
    return np.mean(
        [
            np.mean(percent_absolute_error(y_true, y_pred)),
            np.median(percent_absolute_error(y_true, y_pred)),
            1.0 - rsquared(y_true, y_pred),
        ]
    )
