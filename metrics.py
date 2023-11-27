import numpy as np


def mean_absolute_error(preds, y_true):
    return np.mean(np.abs(preds - y_true))


def mean_squared_error(preds, y_true):
    return np.mean((preds - y_true)**2)


def root_mean_squared_error(preds, y_true):
    return np.sqrt(np.mean((preds - y_true)**2))


def r_squared(preds, y_true):
    mean_true = np.mean(y_true)
    total_variance = np.sum((y_true - mean_true) ** 2)
    residual_variance = np.sum((y_true - preds) ** 2)
    return 1 - (residual_variance / total_variance)
