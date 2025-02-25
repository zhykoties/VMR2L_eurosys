import torch
import numpy as np
import logging

logger = logging.getLogger('VM.Helper')


def accuracy_MAE(predict_all: np.ndarray, gt_all: np.ndarray, missing_value=0):
    zero_index = (gt_all != missing_value)
    diff = np.mean(np.abs(gt_all[zero_index] - predict_all[zero_index]))
    return diff


def accuracy_MSE(predict_all: np.ndarray, gt_all: np.ndarray, missing_value=0):
    zero_index = (gt_all != missing_value)
    diff = np.mean((gt_all[zero_index] - predict_all[zero_index]) ** 2)
    return diff


# for plots
def accuracy_MAPE_(mu: torch.Tensor, labels: torch.Tensor, missing_value=0):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = (labels == missing_value)
    mu[mask] = 1
    labels[mask] = 1
    result = np.average(np.abs(mu - labels) / np.abs(labels), axis=1)
    return result


# for plots
def accuracy_AGG_(mu: torch.Tensor, labels: torch.Tensor, missing_value=0):
    batch_size = mu.shape[0]
    time_step = mu.shape[1] // 30 * 30
    mu = mu[:, :time_step]
    labels = labels[:, :time_step]
    mu[labels == missing_value] = missing_value
    mu = mu.view(batch_size, -1, 30).sum(dim=2)
    labels = labels.view(batch_size, -1, 30).sum(dim=2)

    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = (labels == 0)
    mu[mask] = 1
    labels[mask] = 1
    result = np.average(np.abs((mu - labels) / labels), axis=1)
    return result


# for plots
def accuracy_ROU_(rou: float, rou_pred: torch.Tensor, labels: torch.Tensor, missing_value=0):
    rou_pred = rou_pred.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == missing_value
    rou_pred[mask] = 0.
    labels[mask] = 0.

    abs_diff = np.abs(labels - rou_pred)
    abs_diff_1 = abs_diff.copy()
    abs_diff_1[labels < rou_pred] = 0.
    abs_diff_2 = abs_diff.copy()
    abs_diff_2[labels >= rou_pred] = 0.

    numerator = 2 * (rou * np.sum(abs_diff_1, axis=1) + (1 - rou) * np.sum(abs_diff_2, axis=1))
    denominator = np.sum(np.abs(labels), axis=1)

    mask2 = (denominator == 0)
    denominator[mask2] = 1
    result = numerator / denominator
    result[mask2] = -1
    return result
