############################
# imports
############################
import pdb

import tensorflow as tf
import datetime
import numpy as np
from scipy.sparse.linalg import eigs
from sklearn.metrics import mean_squared_error, mean_absolute_error

############################
# functions
############################


def evaluation(y, y_hat, output_stats):
    '''
    Evaluation function: interface to calculate MAPE, MAE and RMSE between ground truth and prediction.
    :param y: torch.Tensor, ground truth, y.shape = (batch_size, 1, 1, n_nodes)
    :param y_: torch.Tensor, prediction, y_hat.shape = (batch_size, 1, 1, n_nodes)
    :param output_stats: dict, paras of z-scores (mean & std).
    :return: np.ndarray, averaged metric values.
    '''
    # only works for the case of 1 output feature
    
    # .numpy() moves data to CPU. squeeze() removes extra dimensions
    v = z_score_inverse(y, output_stats["mean"], output_stats["std"]).numpy().squeeze()
    v_hat = z_score_inverse(y_hat, output_stats["mean"], output_stats["std"]).numpy().squeeze()
    
    # Handle the case where a batch might have only 1 sample
    if v.ndim == 1:
        v = v.reshape(1, -1)
    if v_hat.ndim == 1:
        v_hat = v_hat.reshape(1, -1)

    # --- FIXED for older scikit-learn ---
    # 1. Calculate Mean Squared Error
    mse = mean_squared_error(v, v_hat)
    # 2. Take the square root to get RMSE
    rmse = np.sqrt(mse)
    # ---
    
    mae = mean_absolute_error(v, v_hat)

    return rmse, mae


def MAPE(y, y_hat):
    """mean absolute percent error
    """
    y_non_zero = y[y != 0]
    y_hat_non_zero = y_hat[y != 0]
    
    if len(y_non_zero) == 0:
        return 0.0 # Avoid division by zero if all actuals are zero
        
    return np.mean(np.abs((y_non_zero - y_hat_non_zero) / y_non_zero)) * 100


def z_score(x, mean, std):
    '''
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    '''
    return (x - mean) / std


def z_score_inverse(x, mean, std):
    '''
    The inverse of function z_score().
    '''
    return (x * std) + mean