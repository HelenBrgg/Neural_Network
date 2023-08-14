import torch


def mean_absolute_error(preds, y_true):
    return torch.mean(torch.abs(preds - y_true))


def mean_squared_error(preds, y_true):
    return torch.mean((preds - y_true)**2)


def root_mean_squared_error(preds, y_true):
    return torch.sqrt(torch.mean((preds - y_true)**2))


def r_squared(preds, y_true):
    mean_true = torch.mean(y_true)
    total_variance = torch.sum((y_true - mean_true) ** 2)
    explained_variance = torch.sum((preds - mean_true) ** 2)
    return 1 - (explained_variance / total_variance)
